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
        /// Delay of the core channel QMF input in timeslots.
        /// </summary>
        const int inputDelay = 10;

        /// <summary>
        /// Real coefficients of the surround band-0 complex FIR, ordered from oldest to newest input.
        /// </summary>
        static readonly float[] surroundFilterReal = CreateSymmetricKernel(new[] {
            0.0013996040215715766f, 0.003839150769636035f, 0.007512642536312342f,
            0.012419373728334904f, 0.018367428332567215f, 0.0249701626598835f,
            0.03167900815606117f, 0.03785000368952751f, 0.04283412545919418f,
            0.04607561603188515f, 0.047200120985507965f
        });

        /// <summary>
        /// Imaginary coefficients of the surround band-0 complex FIR, ordered from oldest to newest input.
        /// </summary>
        static readonly float[] surroundFilterImaginary = CreateSymmetricKernel(new[] {
            -0.0006242550443857908f, -0.0019234686624258757f, -0.0042654648423194885f,
            -0.008168308064341545f, -0.014327201060950756f, -0.023759860545396805f,
            -0.03757232800126076f, -0.05577569454908371f, -0.07568276673555374f,
            -0.09172472357749939f, -0.5979374051094055f
        });

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
        /// Zero-initialized raw QMF history for L, R, C, Ls, and Rs in input matrix order, retained across frames.
        /// </summary>
        readonly (float[] real, float[] imaginary)[] inputHistory = new (float[], float[])[5];

        /// <summary>
        /// Next history timeslot to overwrite, independent of the frame-relative timeslot.
        /// </summary>
        int inputHistoryPosition;

        /// <summary>
        /// Recycled QMFB operation arrays.
        /// </summary>
        readonly (float[] real, float[] imaginary)[] qmfbCache;

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
            for (int ch = 0; ch < inputHistory.Length; ++ch) {
                int historyLength = surroundFilterReal.Length * QuadratureMirrorFilterBank.subbands;
                inputHistory[ch] = (new float[historyLength], new float[historyLength]);
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
                    fixed (float* pInput = input[ch]) {
                        if (!mono) {
                            results[ch] = converters[ch].ProcessForward(pInput);
                        } else {
                            results[ch] = converters[ch].ProcessForward_Mono(pInput);
                        }
                    }
                    if (Interlocked.Decrement(ref runs) == 0) {
                        taskWaiter.Set();
                    }
                }, ch);
            }
            taskWaiter.Wait();

            if (joc.DownmixConfig == 3) {
                PreprocessInputs(joc.ChannelCount);
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
        /// Mirrors the unique taps, including the center tap, into a complete symmetric kernel.
        /// </summary>
        static float[] CreateSymmetricKernel(float[] uniqueTaps) {
            float[] kernel = new float[2 * uniqueTaps.Length - 1];
            for (int tap = 0; tap < uniqueTaps.Length; ++tap) {
                kernel[tap] = kernel[kernel.Length - 1 - tap] = uniqueTaps[tap];
            }
            return kernel;
        }

        /// <summary>
        /// Prepares the core channel QMF inputs once for all object mixing paths.
        /// </summary>
        void PreprocessInputs(int channels) {
            int subbands = QuadratureMirrorFilterBank.subbands;
            int historyLength = surroundFilterReal.Length;
            int inputOffset = inputHistoryPosition * subbands;
            int delayedPosition = inputHistoryPosition - inputDelay;
            if (delayedPosition < 0) {
                delayedPosition += historyLength;
            }
            int delayedOffset = delayedPosition * subbands;

            for (int ch = 0, count = Math.Min(channels, inputHistory.Length); ch < count; ++ch) {
                (float[] real, float[] imaginary) = results[ch];
                (float[] real, float[] imaginary) history = inputHistory[ch];

                // Preserve raw QMF samples before modifying the recycled forward output.
                Array.Copy(real, 0, history.real, inputOffset, subbands);
                Array.Copy(imaginary, 0, history.imaginary, inputOffset, subbands);

                if (ch < 3) { // L, R, C: X[t - 10]
                    Array.Copy(history.real, delayedOffset, real, 0, subbands);
                    Array.Copy(history.imaginary, delayedOffset, imaginary, 0, subbands);
                } else { // Ls, Rs
                    for (int sb = 1; sb < subbands; ++sb) {
                        real[sb] = history.imaginary[delayedOffset + sb];
                        imaginary[sb] = -history.real[delayedOffset + sb];
                    }

                    // Band 0: 2 * sum(h[k] * X[t - 20 + k]), without an additional delay or rotation.
                    float sumReal = 0, sumImaginary = 0;
                    // The next ring entry is the oldest sample, X[t - 20].
                    int sampleOffset = inputOffset + subbands;
                    for (int tap = 0; tap < historyLength; ++tap) {
                        if (sampleOffset == history.real.Length) {
                            sampleOffset = 0;
                        }
                        float sampleReal = history.real[sampleOffset];
                        float sampleImaginary = history.imaginary[sampleOffset];
                        sumReal += surroundFilterReal[tap] * sampleReal - surroundFilterImaginary[tap] * sampleImaginary;
                        sumImaginary += surroundFilterReal[tap] * sampleImaginary + surroundFilterImaginary[tap] * sampleReal;
                        sampleOffset += subbands;
                    }
                    real[0] = 2 * sumReal;
                    imaginary[0] = 2 * sumImaginary;
                }
            }

            if (++inputHistoryPosition == historyLength) {
                inputHistoryPosition = 0;
            }
        }

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