using System;
using System.Runtime.CompilerServices;
using System.Threading;

using Cavern.Utilities;

namespace Cavern.Format.Decoders.EnhancedAC3 {
    /// <summary>
    /// Converts a channel-based audio stream and JOC to object output samples.
    /// </summary>
    class JointObjectCodingApplier : IDisposable {
        /// <summary>
        /// Cavern is run by a Mono runtime, use functions optimized for that.
        /// </summary>
        readonly bool mono;

        /// <summary>
        /// Length of an AC-3 frame.
        /// </summary>
        readonly int frameSize;

        /// <summary>
        /// Recycled timeslot object output arrays.
        /// </summary>
        readonly float[][] timeslotCache;

        /// <summary>
        /// Recycled forward transformation result holder.
        /// </summary>
        readonly (float[] real, float[] imaginary)[] results;

        /// <summary>
        /// Recycled QMFB operation arrays.
        /// </summary>
        readonly (float[] real, float[] imaginary)[] qmfbCache;


        /// <summary>
        /// Experimental JOC input delay, in QMF timeslots. This is deliberately
        /// applied after analysis QMF and before the object mixing matrix.
        /// </summary>
        const int jocQmfDelaySlots = 10;

        /// <summary>
        /// Per-input-channel complex QMF delay line for the delay experiment.
        /// </summary>
        readonly float[][][] jocQmfDelayReal, jocQmfDelayImaginary;

        /// <summary>
        /// Current slot in the 10-timeslot JOC QMF delay line.
        /// </summary>
        int jocQmfDelayPosition;

        /// <summary>
        /// Used for waiting while started tasks work.
        /// </summary>
        readonly ManualResetEventSlim taskWaiter = new ManualResetEventSlim(false);

        /// <summary>
        /// Recycled QMFB transform objects.
        /// </summary>
        readonly QuadratureMirrorFilterBank[] converters;

        /// <summary>
        /// Channels to objects matrix.
        /// </summary>
        float[][][][] mixMatrix;

        /// <summary>
        /// Next timeslot to read in the current JOC.
        /// </summary>
        int timeslot;

        /// <summary>
        /// Creates a converter from a channel-based audio stream and JOC to object output samples.
        /// </summary>
        public JointObjectCodingApplier(JointObjectCoding joc, int frameSize) {
            mono = CavernAmp.IsMono();
            int maxChannels = JointObjectCodingTables.inputMatrix.Length;
            int objects = joc.ObjectCount;
            this.frameSize = frameSize;

            timeslotCache = new float[objects][];
            results = new (float[], float[])[maxChannels];
            qmfbCache = new (float[], float[])[objects];
            jocQmfDelayReal = new float[maxChannels][][];
            jocQmfDelayImaginary = new float[maxChannels][][];
            for (int ch = 0; ch < maxChannels; ++ch) {
                // 'results' must be independent from the converter's recycled
                // outCache because the current QMF slot is immediately pushed
                // into the delay line while the delayed slot is mixed below.
                results[ch] = (new float[QuadratureMirrorFilterBank.subbands],
                    new float[QuadratureMirrorFilterBank.subbands]);
                jocQmfDelayReal[ch] = new float[jocQmfDelaySlots][];
                jocQmfDelayImaginary[ch] = new float[jocQmfDelaySlots][];
                for (int slot = 0; slot < jocQmfDelaySlots; ++slot) {
                    jocQmfDelayReal[ch][slot] = new float[QuadratureMirrorFilterBank.subbands];
                    jocQmfDelayImaginary[ch][slot] = new float[QuadratureMirrorFilterBank.subbands];
                }
            }
            for (int obj = 0; obj < objects; ++obj) {
                timeslotCache[obj] = new float[QuadratureMirrorFilterBank.subbands];
                qmfbCache[obj] = (new float[QuadratureMirrorFilterBank.subbands], new float[QuadratureMirrorFilterBank.subbands]);
            }

            int converterCount = Math.Max(maxChannels, objects);
            converters = new QuadratureMirrorFilterBank[converterCount];
            for (int i = 0; i < converterCount; ++i) {
                converters[i] = new QuadratureMirrorFilterBank();
            }
        }

        /// <summary>
        /// Gets the audio samples of each object for the next timeslot.
        /// </summary>
        public unsafe float[][] Apply(float[][] input, JointObjectCoding joc) {
            if (timeslot == 0) {
                mixMatrix = joc.GetMixingMatrices(frameSize);
            }

            // Forward transformations
            int runs = joc.ChannelCount;
            taskWaiter.Reset();
            for (int ch = 0; ch < joc.ChannelCount; ++ch) {
                ThreadPool.QueueUserWorkItem(channel => {
                    int ch = (int)channel;
                    (float[] real, float[] imaginary) current;
                    fixed (float* pInput = input[ch]) {
                        if (!mono) {
                            current = converters[ch].ProcessForward(pInput);
                        } else {
                            current = converters[ch].ProcessForward_Mono(pInput);
                        }
                    }

                    // Experimental Dolby-JOC-style preprocessing probe:
                    // delay complex QMF by exactly 10 timeslots (10 * 64 samples)
                    // before multiplying it by mixMatrix[obj][timeslot].
                    float[] delayedReal = jocQmfDelayReal[ch][jocQmfDelayPosition];
                    float[] delayedImaginary = jocQmfDelayImaginary[ch][jocQmfDelayPosition];
                    Array.Copy(delayedReal, results[ch].real, QuadratureMirrorFilterBank.subbands);
                    Array.Copy(delayedImaginary, results[ch].imaginary, QuadratureMirrorFilterBank.subbands);
                    Array.Copy(current.real, delayedReal, QuadratureMirrorFilterBank.subbands);
                    Array.Copy(current.imaginary, delayedImaginary, QuadratureMirrorFilterBank.subbands);
                    if (Interlocked.Decrement(ref runs) == 0) {
                        taskWaiter.Set();
                    }
                }, ch);
            }
            taskWaiter.Wait();

            // Advance once per 64-sample input timeslot, after every channel has
            // read/written the same delay-line position.
            if (++jocQmfDelayPosition == jocQmfDelaySlots) {
                jocQmfDelayPosition = 0;
            }

            // Inverse transformations
            int objects = joc.ObjectCount;
            runs = objects;
            taskWaiter.Reset();
            for (int obj = 0; obj < objects; ++obj) {
                ThreadPool.QueueUserWorkItem(objectIndex => {
                    int obj = (int)objectIndex;
                    if (CavernAmp.Available) {
                        ProcessObject_Amp(joc, obj, mixMatrix[obj][timeslot], joc.Gain);
                    } else if (!mono) {
                        ProcessObject(joc, obj, mixMatrix[obj][timeslot], joc.Gain);
                    } else {
                        ProcessObject_Mono(joc, obj, mixMatrix[obj][timeslot], joc.Gain);
                    }
                    if (Interlocked.Decrement(ref runs) == 0) {
                        taskWaiter.Set();
                    }
                }, obj);
            }
            taskWaiter.Wait();

            if (++timeslot == input.Length) {
                timeslot = 0;
            }
            return timeslotCache;
        }

        /// <summary>
        /// Free up resources used by this object.
        /// </summary>
        public void Dispose() => taskWaiter.Dispose();

        /// <summary>
        /// Mixes channel-based samples by a matrix to the objects.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe void ProcessObject(JointObjectCoding joc, int obj, float[][] mixMatrix, float gain) {
            (float[] real, float[] imaginary) = qmfbCache[obj];
            fixed (float* channelReal = results[0].real, channelImaginary = results[0].imaginary, channelMatrix = mixMatrix[0]) {
                QMath.MultiplyAndSet(channelReal, channelMatrix, real, QuadratureMirrorFilterBank.subbands);
                QMath.MultiplyAndSet(channelImaginary, channelMatrix, imaginary, QuadratureMirrorFilterBank.subbands);
            }
            for (int ch = 1; ch < joc.ChannelCount; ch++) {
                fixed (float* channelReal = results[ch].real, channelImaginary = results[ch].imaginary, channelMatrix = mixMatrix[ch]) {
                    QMath.MultiplyAndAdd(channelReal, channelMatrix, real, QuadratureMirrorFilterBank.subbands);
                    QMath.MultiplyAndAdd(channelImaginary, channelMatrix, imaginary, QuadratureMirrorFilterBank.subbands);
                }
            }
            converters[obj].ProcessInverse(qmfbCache[obj], timeslotCache[obj]);
            if (gain != 1) {
                WaveformUtils.Gain(timeslotCache[obj], gain);
            }
        }

        /// <summary>
        /// Mixes channel-based samples by a matrix to the objects.
        /// This version of the function is faster when <see cref="CavernAmp"/> is available.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe void ProcessObject_Amp(JointObjectCoding joc, int obj, float[][] mixMatrix, float gain) {
            (float[] real, float[] imaginary) = qmfbCache[obj];
            fixed (float* pReal = real, pImaginary = imaginary) {
                fixed (float* channelReal = results[0].real, channelImaginary = results[0].imaginary, channelMatrix = mixMatrix[0]) {
                    CavernAmp.MultiplyAndSet(channelReal, channelMatrix, pReal, QuadratureMirrorFilterBank.subbands);
                    CavernAmp.MultiplyAndSet(channelImaginary, channelMatrix, pImaginary, QuadratureMirrorFilterBank.subbands);
                }
                for (int ch = 1; ch < joc.ChannelCount; ch++) {
                    fixed (float* channelReal = results[ch].real, channelImaginary = results[ch].imaginary, channelMatrix = mixMatrix[ch]) {
                        CavernAmp.MultiplyAndAdd(channelReal, channelMatrix, pReal, QuadratureMirrorFilterBank.subbands);
                        CavernAmp.MultiplyAndAdd(channelImaginary, channelMatrix, pImaginary, QuadratureMirrorFilterBank.subbands);
                    }
                }
            }
            if (!mono) {
                converters[obj].ProcessInverse(qmfbCache[obj], timeslotCache[obj]);
            } else {
                fixed (float* pReal = real, pImaginary = imaginary) {
                    converters[obj].ProcessInverse_Amp(pReal, pImaginary, timeslotCache[obj]);
                }
            }
            if (gain != 1) {
                WaveformUtils.Gain(timeslotCache[obj], gain);
            }
        }

        /// <summary>
        /// Mixes channel-based samples by a matrix to the objects.
        /// This version of the function is faster only in a Mono runtime (like Unity).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe void ProcessObject_Mono(JointObjectCoding joc, int obj, float[][] mixMatrix, float gain) {
            (float[] real, float[] imaginary) = qmfbCache[obj];
            fixed (float* pReal = real, pImaginary = imaginary) {
                fixed (float* channelReal = results[0].real, channelImaginary = results[0].imaginary, channelMatrix = mixMatrix[0]) {
                    QMath.MultiplyAndSet_Mono(channelReal, channelMatrix, pReal, QuadratureMirrorFilterBank.subbands);
                    QMath.MultiplyAndSet_Mono(channelImaginary, channelMatrix, pImaginary, QuadratureMirrorFilterBank.subbands);
                }
                for (int ch = 1; ch < joc.ChannelCount; ch++) {
                    fixed (float* channelReal = results[ch].real, channelImaginary = results[ch].imaginary, channelMatrix = mixMatrix[ch]) {
                        QMath.MultiplyAndAdd_Mono(channelReal, channelMatrix, pReal, QuadratureMirrorFilterBank.subbands);
                        QMath.MultiplyAndAdd_Mono(channelImaginary, channelMatrix, pImaginary, QuadratureMirrorFilterBank.subbands);
                    }
                }
            }
            fixed (float* output = timeslotCache[obj]) {
                converters[obj].ProcessInverse_Mono(qmfbCache[obj], output);
            }
            if (gain != 1) {
                WaveformUtils.Gain(timeslotCache[obj], gain);
            }
        }
    }
}
