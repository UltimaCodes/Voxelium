// Where the cloud layer is, for everything that has to agree about it: the sky, and the sun's path to
// the ground. The layer sits above the highest terrain, so nothing ever pokes into it.
layout(location = 33) uniform float uCloudBias; // extra cover from the weather director, 0..~0.22

const float kCloudBase = 430.0;
const float kCloudTop = 650.0;
const float kCloudMid = 0.5 * (kCloudBase + kCloudTop);
