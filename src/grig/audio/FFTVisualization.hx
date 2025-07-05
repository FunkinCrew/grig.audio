package grig.audio;

/*
 * util.c
 * Copyright 2009-2019 John Lindgren and Michał Lipski
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

class FFTVisualization
{
    private var xscale = new Array<Float>();
    private var fftSize:Int = 512;
    private var sampleRate:Float = 44100.0;
    private var minFreq:Float = 20.0;
    private var maxFreq:Float = 20000.0;

    private function computeLogXScale(bands:Int, fftSize:Int, sampleRate:Float, minFreq:Float, maxFreq:Float):Array<Float> {
        var xscale = new Array<Float>();
        xscale.resize(bands + 1);

        // Calculate logarithmic frequency distribution
        var logMin = Math.log(minFreq);
        var logMax = Math.log(maxFreq);
        var logRange = logMax - logMin;

        for (i in 0...bands + 1) {
            var logFreq = logMin + (logRange * i / bands);
            var freq = Math.exp(logFreq);
            // Convert frequency to FFT bin index
            xscale[i] = freq * fftSize / sampleRate;
        }

        return xscale;
    }

    private function computeFreqBand(freq:Array<Float>, xscale:Array<Float>, band:Int, bands:Int, fftSize:Int):Float {
        var startBin = xscale[band];
        var endBin = xscale[band + 1];
        var maxBin = Std.int(fftSize / 2); // Nyquist limit

        // Clamp to valid bin range
        startBin = FFT.clamp(startBin, 0.0, maxBin - 1);
        endBin = FFT.clamp(endBin, 0.0, maxBin - 1);

        var n:Float = 0.0;
        var binRange = endBin - startBin;

        if (binRange < 1.0) {
            // Interpolate between adjacent bins for sub-bin precision
            var lowerBin = Math.floor(startBin);
            var upperBin = Math.ceil(endBin);

            if (lowerBin == upperBin) {
                // Single bin case
                if (lowerBin < freq.length) {
                    n = freq[lowerBin] * binRange;
                }
            } else {
                // Interpolate between two bins
                var fraction = startBin - lowerBin;
                if (lowerBin < freq.length && upperBin < freq.length) {
                    var v0 = freq[lowerBin];
                    var v1 = freq[upperBin];
                    n = (v0 + (v1 - v0) * fraction) * binRange;
                }
            }
        } else {
            // Multiple bins - accumulate with proper weighting
            var currentBin = Math.floor(startBin);
            var lastBin = Math.floor(endBin);

            // Partial first bin
            if (currentBin < freq.length) {
                var firstWeight = Math.ceil(startBin) - startBin;
                n += freq[currentBin] * firstWeight;
            }
            currentBin++;

            // Full middle bins
            while (currentBin < lastBin && currentBin < freq.length) {
                n += freq[currentBin];
                currentBin++;
            }

            // Partial last bin
            if (lastBin < freq.length && lastBin != Math.floor(startBin)) {
                var lastWeight = endBin - lastBin;
                n += freq[lastBin] * lastWeight;
            }
        }

        // Normalize by frequency range and apply dB conversion
        if (n > 0) {
            return 20 * FFT.log(10, n);
        } else {
            return -120.0; // Very low dB value for silence
        }
    }

    public function new() {}

    public function makeLogGraph(freq:Array<Float>, bands:Int, dbRange:Int, intRange:Int, fftSize:Int = 512, sampleRate:Float = 44100.0, minFreq:Float = 20.0, maxFreq:Float = 20000.0):Array<Int> {
        // Store parameters for reuse
        this.fftSize = fftSize;
        this.sampleRate = sampleRate;
        this.minFreq = minFreq;
        this.maxFreq = maxFreq;

        // Recompute frequency scale if parameters changed
        if (xscale.length != bands + 1) {
            xscale = computeLogXScale(bands, fftSize, sampleRate, minFreq, maxFreq);
        }

        var graph = new Array<Int>();
        graph.resize(bands);
        for (i in 0...bands) {
            var val:Float = computeFreqBand(freq, xscale, i, bands, fftSize);
            #if python
            if (Math.isNaN(val)) {
                graph[i] = 0;
                continue;
            }
            #end
            // scale (-db_range, 0.0) to (0.0, int_range)
            val = (1 + val / dbRange) * intRange;
            graph[i] = FFT.clamp(Std.int(val), 0, intRange);
        }

        return graph;
    }
}
