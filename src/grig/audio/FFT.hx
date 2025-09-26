package grig.audio;

/*
 * fft.c
 * Copyright 2011 John Lindgren
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions, and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the following disclaimer in the documentation
 *    provided with the distribution.
 *
 * This software is provided "as is" and without any warranty, express or
 * implied. In no event shall the authors be liable for any damages arising from
 * the use of this software.
 */

// Original c code copyright 2011 John Lindgren, ported to haxe by Thomas J. Webb 2024

class FFT
{
    private static final TWO_PI:Float = 6.2831853;

    private var hamming = new Array<Float>();   // hamming window, scaled to sum to 1
    private var reversed = new Array<Int>();    // bit-reversal table
    private var roots = new Array<Complex>();   // N-th roots of unity

    private var workingArray = new Array<Complex>();
    private var outputFreq = new Array<Float>();

    private var realArray = new Array<Float>();
    private var imagArray = new Array<Float>();

    private var n:Int;
    private var logN:Int;

    public function new(n:Int = 512) {
        this.n = n;
        logN = Std.int(log(2.0, n));

        // Pre-allocate all arrays
        hamming.resize(n);
        reversed.resize(n);
        roots.resize(Std.int(n / 2));
        workingArray.resize(n);
        outputFreq.resize(Std.int(n / 2));

        realArray.resize(n);
        imagArray.resize(n);

        // Pre-allocate Complex objects in working array
        for (i in 0...n) {
            workingArray[i] = new Complex(0.0, 0.0);
            realArray[i] = 0.0;
            imagArray[i] = 0.0;
        }

        generateTables();
    }

    // This should be moved to a utility class somewhere
    public static function log(base:Float, x:Float):Float {
        return Math.log(x) / Math.log(base);
    }

    @:generic
    public static function clamp<T:Float>(value:T, lower:T, upper:T):T {
        return value < lower ? lower : (value > upper ? upper : value);
    }

    // Reverse the order of the lowest LOGN bits in an integer.
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

    // Generate lookup tables.
    private function generateTables() {
        for (i in 0...n)
            hamming[i] = 1 - 0.85 * Math.cos(i * (TWO_PI / n));
        for (i in 0...reversed.length)
            reversed[i] = bitReverse(i);
        for (i in 0...Std.int(n / 2))
            roots[i] = Complex.exp(new Complex(0, i * (TWO_PI / n)));
    }

    /*
     * Perform the DFT using the Cooley-Tukey algorithm.  At each step s, where
     * s=1..log N (base 2), there are N/(2^s) groups of intertwined butterfly
     * operations.  Each group contains (2^s)/2 butterflies, and each butterfly has
     * a span of (2^s)/2.  The twiddle factors are nth roots of unity where n = 2^s.
     */
    private function doFFT(a:Array<Complex>) {
        var half:Int = 1;
        var inv = Std.int(a.length / 2);

        while (inv > 0) {
            var g:Int = 0;

            while (g < a.length) {
                var b:Int = 0;
                var r:Int = 0;

                while (b < half) {
                    var evenIdx = g + b;
                    var oddIdx = g + half + b;

                    // Cache the even value
                    var evenReal = a[evenIdx].real;
                    var evenImag = a[evenIdx].imag;

                    // Calculate odd * root in-place to avoid allocation
                    var rootReal = roots[r].real;
                    var rootImag = roots[r].imag;
                    var oddReal = a[oddIdx].real;
                    var oddImag = a[oddIdx].imag;

                    // Multiply odd by root
                    var tempReal = oddReal * rootReal - oddImag * rootImag;
                    var tempImag = oddReal * rootImag + oddImag * rootReal;

                    // Butterfly operation - modify in place
                    a[evenIdx].set(evenReal + tempReal, evenImag + tempImag);
                    a[oddIdx].set(evenReal - tempReal, evenImag - tempImag);

                    b++;
                    r += inv;
                }

                g += half << 1;
            }

            half <<= 1;
            inv >>= 1;
        }
    }

    private function doFFTSeparate(real:Array<Float>, imag:Array<Float>) {
        var half:Int = 1;
        var inv = Std.int(n / 2);

        while (inv > 0) {
            for (g in 0...n) {
                if (g % (half << 1) != 0) continue;

                for (b in 0...half) {
                    var evenIdx = g + b;
                    var oddIdx = g + half + b;
                    var rootIdx = b * inv;

                    // Cache values
                    var evenReal = real[evenIdx];
                    var evenImag = imag[evenIdx];
                    var rootReal = roots[rootIdx].real;
                    var rootImag = roots[rootIdx].imag;

                    // Complex multiplication: odd * root
                    var tempReal = real[oddIdx] * rootReal - imag[oddIdx] * rootImag;
                    var tempImag = real[oddIdx] * rootImag + imag[oddIdx] * rootReal;

                    // Butterfly operation
                    real[evenIdx] = evenReal + tempReal;
                    imag[evenIdx] = evenImag + tempImag;
                    real[oddIdx] = evenReal - tempReal;
                    imag[oddIdx] = evenImag - tempImag;
                }
            }

            half <<= 1;
            inv >>= 1;
        }
    }

    // Input is N=512 PCM samples.
    // Output is intensity of frequencies from 1 to N/2=256.
    public function calcFreq(data:Array<Float>):Array<Float> {
        for (i in 0...n) {
            var reversedIdx = reversed[i];
            // workingArray[reversedIdx].set(data[i] * hamming[i], 0.0);
            realArray[reversedIdx] = data[i] * hamming[i];
            imagArray[reversedIdx] = 0.0;
        }

        // doFFT(workingArray);
        doFFTSeparate(realArray, imagArray);


        // Calculate magnitudes without creating intermediate Complex objects
        var halfN = Std.int(n / 2);
        var invN = 1.0 / n;

        for (i in 0...halfN) {
            var real = realArray[1 + i];
            var imag = imagArray[1 + i];

            // var real = workingArray[1 + i].real;
            // var imag = workingArray[1 + i].imag;
            outputFreq[i] = 2 * Math.sqrt(real * real + imag * imag) * invN;
        }

        // Handle the Nyquist frequency (not doubled)
        var nyquistReal = realArray[halfN];
        var nyquistImag = imagArray[halfN];
        outputFreq[halfN - 1] = Math.sqrt(nyquistReal * nyquistReal + nyquistImag * nyquistImag) * invN;

        return outputFreq;
    }
}
