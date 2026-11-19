package com.example.vision

import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import kotlin.math.min

object FaceMeshTriangulation {

    const val LANDMARK_COUNT = 478

    val innerLipRing: IntArray = intArrayOf(
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    )

    val faceOvalSet: Set<Int> = setOf(
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    )

    val leftEyeAperture: Set<Int> = setOf(
        33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
    )

    val rightEyeAperture: Set<Int> = setOf(
        362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
    )

    val foreheadRimSet: Set<Int> = setOf(
        10, 338, 297, 332, 284, 251, 389, 21, 54, 103, 67, 109,
        151, 337, 299, 333, 298, 301, 108, 69, 104, 68, 71,
    )

    val skinSampleIndices: IntArray = intArrayOf(50, 280, 101, 330, 195, 4)

    private val innerLipSet: Set<Int> = innerLipRing.toSet()

    private class Topology(
        val triangles: ShortArray,
        val vertexAlphas: FloatArray,
    )

    private val topology: Topology by lazy {
        val adj = HashMap<Int, MutableSet<Int>>(512)
        for (edge in FaceLandmarker.FACE_LANDMARKS_TESSELATION) {
            val u = edge.start()
            val v = edge.end()
            if (u == v) continue
            adj.getOrPut(u) { HashSet() }.add(v)
            adj.getOrPut(v) { HashSet() }.add(u)
        }

        val distFromOval = IntArray(LANDMARK_COUNT) { 99 }
        val queue = ArrayDeque<Int>(LANDMARK_COUNT)
        for (idx in faceOvalSet) {
            if (idx in 0 until LANDMARK_COUNT) {
                distFromOval[idx] = 0
                queue.addLast(idx)
            }
        }
        while (queue.isNotEmpty()) {
            val curr = queue.removeFirst()
            val nextDist = distFromOval[curr] + 1
            val neighbors = adj[curr] ?: continue
            for (nb in neighbors) {
                if (nb in 0 until LANDMARK_COUNT && distFromOval[nb] > nextDist) {
                    distFromOval[nb] = nextDist
                    queue.addLast(nb)
                }
            }
        }

        val alphas = FloatArray(LANDMARK_COUNT)
        for (i in 0 until LANDMARK_COUNT) {
            var alpha = when (distFromOval[i]) {
                0 -> 0.0f
                1 -> 0.16f
                2 -> 0.52f
                3 -> 0.85f
                else -> 1.0f
            }
            if (foreheadRimSet.contains(i)) {
                alpha *= 0.42f
            }
            if (leftEyeAperture.contains(i) || rightEyeAperture.contains(i)) {
                alpha = min(alpha, 0.12f)
            }
            alphas[i] = alpha
        }

        val indices = ArrayList<Short>(2700)
        for ((u, uNeighbors) in adj) {
            for (v in uNeighbors) {
                if (v <= u) continue
                val vNeighbors = adj[v] ?: continue
                for (w in vNeighbors) {
                    if (w <= v) continue
                    if (!uNeighbors.contains(w)) continue
                    if (innerLipSet.contains(u) && innerLipSet.contains(v) && innerLipSet.contains(w)) {
                        continue
                    }
                    if ((leftEyeAperture.contains(u) && leftEyeAperture.contains(v) && leftEyeAperture.contains(w)) ||
                        (rightEyeAperture.contains(u) && rightEyeAperture.contains(v) && rightEyeAperture.contains(w))
                    ) {
                        continue
                    }
                    indices.add(u.toShort())
                    indices.add(v.toShort())
                    indices.add(w.toShort())
                }
            }
        }

        Topology(indices.toShortArray(), alphas)
    }

    val triangleIndices: ShortArray
        get() = topology.triangles

    val vertexAlphas: FloatArray
        get() = topology.vertexAlphas
}
