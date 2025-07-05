package grig.audio;

import haxe.ds.Vector;

class FFT
{
    private static final TWO_PI:Float = 6.2831853;

    // Pre-allocated working vectors - reused across all FFT calls
    // TODO: Implement configurable windowing (Hamming, Blackman, Hann, etc.)
    private var windowTables:Vector<Float>;
    private var reversed:Vector<Int>;
    private var twiddleReal:Vector<Float>;
    private var twiddleImag:Vector<Float>;

    // Working arrays for separate real/imaginary data
    private var workingReal:Vector<Float>;
    private var workingImag:Vector<Float>;

    // Output frequency magnitudes
    private var outputFreq:Vector<Float>;

    private var n:Int;
    private var logN:Int;
    
    // Prime factorization of N (radix sequence)
    private var radixFactors:Vector<Int>;
    private var numFactors:Int;

    public function new(n:Int = 512) {
        // Round to nearest valid FFT size if needed
        var validN = roundToNearestPowerOf2(n);
        if (validN != n) {
            trace('Warning: FFT size $n is not a power of 2. Rounded to $validN');
        }
        
        this.n = validN;
        logN = Std.int(log(2.0, validN));

        // Validate and factorize N (should always succeed now)
        if (!validateAndFactorize(validN)) {
            throw "Internal error: failed to factorize rounded FFT size";
        }

        // Allocate all vectors once during construction
        windowTables = new Vector<Float>(n);
        reversed = new Vector<Int>(n);
        twiddleReal = new Vector<Float>(n);
        twiddleImag = new Vector<Float>(n);

        workingReal = new Vector<Float>(n);
        workingImag = new Vector<Float>(n);

        outputFreq = new Vector<Float>(Std.int(n / 2));

        generateTables();
    }

    public static function log(base:Float, x:Float):Float {
        return Math.log(x) / Math.log(base);
    }

    private static function roundToNearestPowerOf2(n:Int):Int {
        if (n <= 0) return 1;
        if ((n & (n - 1)) == 0) return n; // Already power of 2
        
        // Find next power of 2
        var nextPow2 = 1;
        while (nextPow2 < n) {
            nextPow2 <<= 1;
        }
        
        // Find previous power of 2
        var prevPow2 = nextPow2 >> 1;
        
        // Return closest one
        return (n - prevPow2) < (nextPow2 - n) ? prevPow2 : nextPow2;
    }

    @:generic
    public static function clamp<T:Float>(value:T, lower:T, upper:T):T {
        return value < lower ? lower : (value > upper ? upper : value);
    }

    private function bitReverse(x:Int):Int {
        var y:Int = 0;
        var i:Int = logN;

        while (i > 0) {
            y = (y << 1) | (x & 1);
            x >>= 1;
            i--;
        }

        return y;
    }

    private function generateTables() {
        // Generate Blackman window (matches Web Audio API AnalyzerNode)
        generateBlackmanWindow();

        // Generate bit reversal table
        for (i in 0...n)
            reversed[i] = bitReverse(i);

        // Generate twiddle factors
        for (i in 0...n) {
            var angle = -TWO_PI * i / n;
            twiddleReal[i] = Math.cos(angle);
            twiddleImag[i] = Math.sin(angle);
        }
    }

    private function generateBlackmanWindow() {
        // Blackman window: w[n] = 0.42 - 0.5*cos(2πn/N) + 0.08*cos(4πn/N)
        for (i in 0...n) {
            var factor = i / (n - 1);
            windowTables[i] = 0.42 - 0.5 * Math.cos(TWO_PI * factor) + 0.08 * Math.cos(4 * Math.PI * factor);
        }
    }

    /**
     * ZERO-ALLOCATION Radix-4 FFT using separate real/imaginary vectors
     */
    private function doFFTRadix4Optimized() {
        var groupSize = 4;

        while (groupSize <= n) {
            var numGroups = Std.int(n / groupSize);
            var quarterSize = Std.int(groupSize / 4);
            var twiddleStep = Std.int(n / groupSize);

            for (group in 0...numGroups) {
                var groupStart = group * groupSize;

                for (k in 0...quarterSize) {
                    var i0 = groupStart + k;
                    var i1 = i0 + quarterSize;
                    var i2 = i1 + quarterSize;
                    var i3 = i2 + quarterSize;

                    // Calculate twiddle factor indices
                    var w1_idx = (k * twiddleStep) % n;
                    var w2_idx = (2 * k * twiddleStep) % n;
                    var w3_idx = (3 * k * twiddleStep) % n;

                    // Optimized 4-point butterfly with separate real/imaginary
                    butterfly4PointOptimized(i0, i1, i2, i3, w1_idx, w2_idx, w3_idx);
                }
            }

            groupSize *= 4;
        }
    }

    /**
     * Zero-allocation 4-point butterfly operation
     * Operates directly on workingReal/workingImag vectors
     */
    private inline function butterfly4PointOptimized(
        i0:Int, i1:Int, i2:Int, i3:Int,
        w1_idx:Int, w2_idx:Int, w3_idx:Int
    ) {
        // Load input values
        var x0r = workingReal[i0];
        var x0i = workingImag[i0];

        // Apply twiddle factors to x1, x2, x3
        // x1 = workingData[i1] * twiddle1
        var x1r_raw = workingReal[i1];
        var x1i_raw = workingImag[i1];
        var tw1r = twiddleReal[w1_idx];
        var tw1i = twiddleImag[w1_idx];
        var x1r = x1r_raw * tw1r - x1i_raw * tw1i;
        var x1i = x1r_raw * tw1i + x1i_raw * tw1r;

        // x2 = workingData[i2] * twiddle2
        var x2r_raw = workingReal[i2];
        var x2i_raw = workingImag[i2];
        var tw2r = twiddleReal[w2_idx];
        var tw2i = twiddleImag[w2_idx];
        var x2r = x2r_raw * tw2r - x2i_raw * tw2i;
        var x2i = x2r_raw * tw2i + x2i_raw * tw2r;

        // x3 = workingData[i3] * twiddle3
        var x3r_raw = workingReal[i3];
        var x3i_raw = workingImag[i3];
        var tw3r = twiddleReal[w3_idx];
        var tw3i = twiddleImag[w3_idx];
        var x3r = x3r_raw * tw3r - x3i_raw * tw3i;
        var x3i = x3r_raw * tw3i + x3i_raw * tw3r;

        // Compute intermediate values for 4-point DFT
        var t0r = x0r + x2r;  // (x0 + x2).real
        var t0i = x0i + x2i;  // (x0 + x2).imag
        var t1r = x0r - x2r;  // (x0 - x2).real
        var t1i = x0i - x2i;  // (x0 - x2).imag
        var t2r = x1r + x3r;  // (x1 + x3).real
        var t2i = x1i + x3i;  // (x1 + x3).imag
        var t3r = x1r - x3r;  // (x1 - x3).real
        var t3i = x1i - x3i;  // (x1 - x3).imag

        // Apply j multiplication: j * (a + jb) = -b + ja
        var jt3r = -t3i;  // j * t3.real = -t3.imag
        var jt3i = t3r;   // j * t3.imag = t3.real

        // Final 4-point DFT butterfly outputs
        workingReal[i0] = t0r + t2r;        // X[k]
        workingImag[i0] = t0i + t2i;
        workingReal[i1] = t1r - jt3r;       // X[k + N/4]
        workingImag[i1] = t1i - jt3i;
        workingReal[i2] = t0r - t2r;        // X[k + N/2]
        workingImag[i2] = t0i - t2i;
        workingReal[i3] = t1r + jt3r;       // X[k + 3N/4]
        workingImag[i3] = t1i + jt3i;
    }

    /**
     * Validates N and computes prime factorization into radix-2 and radix-4 components
     */
    private function validateAndFactorize(n:Int):Bool {
        if (n <= 0 || (n & (n - 1)) != 0) {
            return false; // N must be power of 2
        }

        // Compute factorization: prefer radix-4 over radix-2
        var factors = new Array<Int>();
        var remaining = n;
        
        // Factor out as many radix-4 stages as possible
        while (remaining % 4 == 0) {
            factors.push(4);
            remaining = Std.int(remaining / 4);
        }
        
        // Factor out remaining radix-2 stages
        while (remaining % 2 == 0) {
            factors.push(2);
            remaining = Std.int(remaining / 2);
        }
        
        if (remaining != 1) {
            return false; // N has factors other than 2 and 4
        }
        
        // Store factorization for use in FFT computation
        numFactors = factors.length;
        radixFactors = new Vector<Int>(numFactors);
        for (i in 0...numFactors) {
            radixFactors[i] = factors[i];
        }
        
        return true;
    }

    /**
     * Proper mixed-radix FFT implementation with correct stage ordering
     */
    private function doFFTMixedRadix() {
        var currentSize = 1;
        
        // Apply stages in correct order based on factorization
        for (stage in 0...numFactors) {
            var radix = radixFactors[stage];
            currentSize *= radix;
            
            if (radix == 4) {
                applyRadix4Stage(currentSize);
            } else if (radix == 2) {
                applyRadix2Stage(currentSize);
            }
        }
    }

    private function applyRadix4Stage(stageSize:Int) {
        var numGroups = Std.int(n / stageSize);
        var quarterSize = Std.int(stageSize / 4);
        var twiddleStep = Std.int(n / stageSize);

        for (group in 0...numGroups) {
            var groupStart = group * stageSize;

            for (k in 0...quarterSize) {
                var i0 = groupStart + k;
                var i1 = i0 + quarterSize;
                var i2 = i1 + quarterSize;
                var i3 = i2 + quarterSize;

                var w1_idx = (k * twiddleStep) % n;
                var w2_idx = (2 * k * twiddleStep) % n;
                var w3_idx = (3 * k * twiddleStep) % n;

                butterfly4PointOptimized(i0, i1, i2, i3, w1_idx, w2_idx, w3_idx);
            }
        }
    }

    private function applyRadix2Stage(stageSize:Int) {
        var numGroups = Std.int(n / stageSize);
        var halfSize = Std.int(stageSize / 2);
        var twiddleStep = Std.int(n / stageSize);

        for (group in 0...numGroups) {
            var groupStart = group * stageSize;

            for (k in 0...halfSize) {
                var i0 = groupStart + k;
                var i1 = i0 + halfSize;

                var w_idx = (k * twiddleStep) % n;
                butterfly2PointOptimized(i0, i1, w_idx);
            }
        }
    }

    /**
     * Optimized 2-point butterfly for mixed-radix
     */
    private inline function butterfly2PointOptimized(i0:Int, i1:Int, w_idx:Int) {
        // temp = workingData[i1] * twiddle
        var x1r = workingReal[i1];
        var x1i = workingImag[i1];
        var twr = twiddleReal[w_idx];
        var twi = twiddleImag[w_idx];

        var tempr = x1r * twr - x1i * twi;
        var tempi = x1r * twi + x1i * twr;

        // Butterfly operation
        var x0r = workingReal[i0];
        var x0i = workingImag[i0];

        workingReal[i1] = x0r - tempr;
        workingImag[i1] = x0i - tempi;
        workingReal[i0] = x0r + tempr;
        workingImag[i0] = x0i + tempi;
    }

    /**
     * Main FFT computation - zero allocations during execution
     */
    public function calcFreq(data:Array<Float>):Array<Float> {

        return calcFreqVector(data).toArray();
    }

    /**
     * High-performance version that returns Vector directly
     */
    public function calcFreqVector(data:Array<Float>):Vector<Float> {
        var dataLength = data.length;
        for (i in 0...n) {
            var reversedIdx = reversed[i];
            var inputValue = (i < dataLength) ? data[i] : 0.0;
            workingReal[reversedIdx] = inputValue * windowTables[i];
            workingImag[reversedIdx] = 0.0;
        }

        if ((logN % 2) == 0) {
            doFFTRadix4Optimized();
        } else {
            doFFTMixedRadix();
        }

        var halfN = Std.int(n / 2);
        var invN = 1.0 / n;

        for (i in 0...halfN) {
            var real = workingReal[i];
            var imag = workingImag[i];
            if (i == 0) {
                // DC component (real-only)
                outputFreq[i] = Math.sqrt(real * real + imag * imag) * invN;
            } else {
                // Other frequency components (multiply by 2 for single-sided spectrum)
                outputFreq[i] = 2 * Math.sqrt(real * real + imag * imag) * invN;
            }
        }

        return outputFreq;
    }

    /**
     * Bulk magnitude calculation - optimized for when you need multiple magnitudes
     */
    public inline function getMagnitude(index:Int):Float {
        var real = workingReal[index];
        var imag = workingImag[index];
        return Math.sqrt(real * real + imag * imag);
    }

    /**
     * Direct access to working data for advanced use cases
     */
    public inline function getReal(index:Int):Float {
        return workingReal[index];
    }

    public inline function getImag(index:Int):Float {
        return workingImag[index];
    }
}
