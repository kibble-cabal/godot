#ifndef POINT_SAMPLER_H
#define POINT_SAMPLER_H

#include "core/object/ref_counted.h"
#include "scene/resources/mesh.h"

class PointSampler : public RefCounted {
	GDCLASS(PointSampler, RefCounted);

	AABB aabb;
	Vector<Face3> faces;
	Vector<float> cumulative_areas;

protected:
	static void _bind_methods();

public:
	void init(const Ref<Mesh> &mesh);
	Vector<Vector3> random(int amount, uint64_t seed);
	Vector<Vector3> poisson(float radius, float density, uint64_t seed);

	PointSampler();
};

#endif // POINT_SAMPLER_H