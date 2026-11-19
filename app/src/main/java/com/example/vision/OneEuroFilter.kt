package com.example.vision

import kotlin.math.PI
import kotlin.math.abs

private class LowPassFilter(private var alpha: Double) {
    private var y = 0.0
    private var s = 0.0
    private var initialized = false

    fun filter(value: Double, newAlpha: Double): Double {
        alpha = newAlpha
        if (!initialized) {
            s = value
            initialized = true
        } else {
            s = alpha * value + (1.0 - alpha) * s
        }
        y = value
        return s
    }

    fun lastRaw(): Double = y

    fun reset() {
        initialized = false
    }
}

class OneEuroFilter(
    private var freq: Double = 30.0,
    private var minCutoff: Double = 1.0,
    private var beta: Double = 0.05,
    private val dCutoff: Double = 1.0,
) {
    private val xFilter = LowPassFilter(computeAlpha(minCutoff))
    private val dxFilter = LowPassFilter(computeAlpha(dCutoff))
    private var lastTimeSec: Double? = null

    fun updateParams(minCutoff: Double, beta: Double) {
        this.minCutoff = minCutoff
        this.beta = beta
    }

    private fun computeAlpha(cutoff: Double): Double {
        val te = 1.0 / freq.coerceAtLeast(1.0)
        val tau = 1.0 / (2.0 * PI * cutoff.coerceAtLeast(0.01))
        return 1.0 / (1.0 + tau / te)
    }

    fun filter(value: Double, timestampSec: Double): Double {
        val prevTime = lastTimeSec
        val hasPrev = prevTime != null
        if (prevTime != null && timestampSec > prevTime) {
            freq = (1.0 / (timestampSec - prevTime)).coerceIn(1.0, 120.0)
        }
        lastTimeSec = timestampSec

        val dx = if (!hasPrev) 0.0 else (value - xFilter.lastRaw()) * freq
        val edx = dxFilter.filter(dx, computeAlpha(dCutoff))
        val cutoff = minCutoff + beta * abs(edx)
        return xFilter.filter(value, computeAlpha(cutoff))
    }

    fun reset() {
        lastTimeSec = null
        xFilter.reset()
        dxFilter.reset()
    }
}

class LandmarkOneEuroSmoother {
    private var filtersX: Array<OneEuroFilter> = emptyArray()
    private var filtersY: Array<OneEuroFilter> = emptyArray()
    private val jawFilter = OneEuroFilter(30.0, 2.0, 0.05, 1.0)

    fun smooth(
        rawPointsXY: FloatArray,
        landmarkCount: Int,
        timestampMs: Long,
        smoothingStrength: Float,
    ): FloatArray {
        val tSec = timestampMs / 1000.0
        val minCutoff = 4.2 - smoothingStrength * 2.8
        val beta = 0.65 - smoothingStrength * 0.35

        if (filtersX.size != landmarkCount) {
            filtersX = Array(landmarkCount) { OneEuroFilter(30.0, minCutoff, beta, 1.0) }
            filtersY = Array(landmarkCount) { OneEuroFilter(30.0, minCutoff, beta, 1.0) }
        }

        val out = FloatArray(landmarkCount * 2)
        for (i in 0 until landmarkCount) {
            filtersX[i].updateParams(minCutoff, beta)
            filtersY[i].updateParams(minCutoff, beta)
            out[i * 2] = filtersX[i].filter(rawPointsXY[i * 2].toDouble(), tSec).toFloat()
            out[i * 2 + 1] = filtersY[i].filter(rawPointsXY[i * 2 + 1].toDouble(), tSec).toFloat()
        }
        return out
    }

    fun smoothJawOpen(rawJawOpen: Float, timestampMs: Long): Float {
        return jawFilter.filter(rawJawOpen.toDouble(), timestampMs / 1000.0).toFloat().coerceIn(0f, 1f)
    }

    fun reset() {
        filtersX.forEach { it.reset() }
        filtersY.forEach { it.reset() }
        jawFilter.reset()
    }
}
