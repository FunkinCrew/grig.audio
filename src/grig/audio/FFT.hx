package grig.audio;

import haxe.ds.Vector;

class FFT
{
    private static final TWO_PI:Float = 6.2831853;

    // Pre-allocated working vectors - reused across all FFT calls
    private var hamming:Vector<Float>;
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

    public function new(n:Int = 512) {
        this.n = n;
        logN = Std.int(log(2.0, n));

        // Allocate all vectors once during construction
        hamming = new Vector<Float>(n);
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
        // Generate Hamming window
        for (i in 0...n)
            hamming[i] = 1 - 0.85 * Math.cos(i * (TWO_PI / n));

        // Generate bit reversal table
        for (i in 0...n)
            reversed[i] = bitReverse(i);
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
     * Mixed-radix implementation for when N is not a pure power of 4
     */
    private function doFFTMixedRadix() {
        // Apply all possible radix-4 stages first
        var currentSize = 4;
        while (currentSize <= n && (n % currentSize) == 0) {
            applyRadix4Stage(currentSize);
            currentSize *= 4;
        }

        // Apply remaining radix-2 stages
        currentSize = 2;
        while (currentSize <= n) {
            if ((n % currentSize) == 0) {
                var stageN = Std.int(n / currentSize);
                if ((stageN % 2) == 1) {  // Only if this stage is needed
                    applyRadix2Stage(currentSize);
                }
            }
            currentSize *= 2;
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
        for (i in 0...n) {
            var reversedIdx = reversed[i];
            workingReal[reversedIdx] = data[i] * hamming[i];
            workingImag[reversedIdx] = 0.0;
        }

        if ((logN % 2) == 0) {
            doFFTRadix4Optimized();
        } else {
            doFFTMixedRadix();
        }

        var halfN = Std.int(n / 2);
        var invN = 1.0 / n;

        for (i in 0...halfN - 1) {
            var real = workingReal[1 + i];
            var imag = workingImag[1 + i];
            outputFreq[i] = 2 * Math.sqrt(real * real + imag * imag) * invN;
        }

        var nyquistReal = workingReal[halfN];
        var nyquistImag = workingImag[halfN];
        outputFreq[halfN - 1] = Math.sqrt(nyquistReal * nyquistReal + nyquistImag * nyquistImag) * invN;

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
