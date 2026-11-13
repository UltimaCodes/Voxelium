// The weather map, shared by the cloud march, the cloud shadows, the sun's shafts and the clouds in the
// water, so all of them agree with the clouds you can see. World anchored, and the wind moves the map,
// not the camera. The includer has clouds.glsl first, and may pick the unit with CLOUD_WEATHER_BINDING.
#ifndef CLOUD_WEATHER_BINDING
#define CLOUD_WEATHER_BINDING 11
#endif
layout(binding = CLOUD_WEATHER_BINDING) uniform sampler2D uCloudWeather;

const float kWeatherSpan = 12288.0;
const vec2 kCloudWind = vec2(2.8, 1.75);

// r how much cloud wants to be here, g how tall it grows
vec2 cloudWeather(vec2 xz, float time) {
    return textureLod(uCloudWeather, (xz + kCloudWind * time) / kWeatherSpan, 0.0).rg;
}

// 0 clear sky, 1 full cover. A storm drops the threshold, so the same map just has more of itself.
float cloudCoverage(float raw) {
    return smoothstep(0.50 - uCloudBias * 1.3, 0.70 - uCloudBias * 0.6, raw);
}
