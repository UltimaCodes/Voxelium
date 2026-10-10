#version 460 core

// wireframe cube edges, vertex-pulled from gl_VertexID (no buffers). corners are
// camera-relative so this pairs with the same proj * viewAtOrigin the world uses.
layout(location = 0) uniform mat4 uViewProj; // 0..3
layout(location = 4) uniform vec3 uMin;      // camera-relative min corner of the voxel

const vec3 kCorner[8] =
    vec3[8](vec3(0, 0, 0), vec3(1, 0, 0), vec3(1, 1, 0), vec3(0, 1, 0), vec3(0, 0, 1),
            vec3(1, 0, 1), vec3(1, 1, 1), vec3(0, 1, 1));
const int kEdge[24] = int[24](0, 1, 1, 2, 2, 3, 3, 0,  // bottom ring
                              4, 5, 5, 6, 6, 7, 7, 4,  // top ring
                              0, 4, 1, 5, 2, 6, 3, 7); // verticals

void main() {
    vec3 p = kCorner[kEdge[gl_VertexID]];
    // inflate a hair from the centre so the near edges win the depth test and the far
    // edges lose it, free hidden-line removal against reversed-Z depth
    p += (p - 0.5) * 0.02;
    gl_Position = uViewProj * vec4(uMin + p, 1.0);
}
