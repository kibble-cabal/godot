#include "point_sampler.h"
#include "core/math/random_number_generator.h"
#include "core/os/time.h"
#include <algorithm>
#include <execution>
#include <vector>

void PointSampler::init(const Ref<Mesh> &mesh) {
	// uint64_t start = Time::get_singleton()->get_ticks_usec();
	this->aabb = mesh->get_aabb();
	this->faces = mesh->get_faces();
	int size = this->faces.size();
	this->cumulative_areas.resize(size);
	float total_area = 0.0;
	for (int i = 0; i < size; i++) {
		Face3 face = this->faces[i];
		float area = face.get_area();
		total_area += area;
		this->cumulative_areas.set(i, total_area);
	}
	// print_line("PointSampler::init ", (Time::get_singleton()->get_ticks_usec() - start) / 1000.0f, "ms");
}

Vector<Vector3> PointSampler::random(int amount, uint64_t seed) {
	// uint64_t start = Time::get_singleton()->get_ticks_usec();
	Vector<Vector3> samples = Vector<Vector3>();
	samples.resize(amount);
	float total_area = this->cumulative_areas[this->cumulative_areas.size() - 1];
	struct TaskData {
		uint64_t seed;
		float total_area;
		Vector<float> *cumulative_areas;
		Vector<Face3> *faces;
		Vector<Vector3> *samples;
	};
	TaskData task_data = {
		seed,
		total_area,
		&this->cumulative_areas,
		&this->faces,
		&samples,
	};
	auto task_id = WorkerThreadPool::get_singleton()->add_native_group_task([](void *userdata, uint32_t idx) {
		auto t = *((TaskData *)userdata);
		RandomPCG rng = RandomPCG(t.seed, idx * 3);
		float r = rng.randf() * t.total_area;
		int j = t.cumulative_areas->bsearch(r, true);
		Face3 face = (*t.faces)[j];
		Vector3 a = face.vertex[0];
		Vector3 b = face.vertex[1];
		Vector3 c = face.vertex[2];
		float r1 = Math::sqrt(rng.randf());
		float r2 = rng.randf();
		// https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
		Vector3 point = (1.0f - r1) * a + r1 * (1.0f - r2) * b + r1 * r2 * c;
		t.samples->set(idx, point);
	},
			&task_data, amount);
	WorkerThreadPool::get_singleton()->wait_for_group_task_completion(task_id);
	// print_line("PointSampler::random ", (Time::get_singleton()->get_ticks_usec() - start) / 1000.0f, "ms");
	return samples;
}

struct Sample {
	Vector3 pos;
	uint32_t cell_id;
};

struct Cell {
	uint32_t cell_id;
	uint32_t start_idx;
	Vector3 sample;
};

struct PhaseCell {
	uint32_t phase_id;
	uint32_t cell_id;
};

inline uint32_t id_from_cell(Vector3i cell, Vector3i grid_size) {
	return cell.x + (cell.y * grid_size.x) + (cell.z * grid_size.x * grid_size.y);
}
inline Vector3i cell_from_id(uint32_t id, Vector3i grid_size) {
	uint32_t x = id % grid_size.x;
	uint32_t y = (id / grid_size.x) % grid_size.y;
	uint32_t z = (id / grid_size.x) / grid_size.y;
	return Vector3i(x, y, z);
}

Vector<Vector3> PointSampler::poisson(float radius, float density, uint64_t seed) {
	float total_area = this->cumulative_areas[this->cumulative_areas.size() - 1];
	float expected_samples = (2.0 * total_area * 0.75 * 0.75) / (Math::sqrt(3.0f) * radius * radius);
	uint32_t oversample_size = (uint32_t)(density * expected_samples);
	float cell_size = 2.0 * radius / Math::sqrt(3.0f);
	Vector3i grid_min = (this->aabb.get_position() / cell_size).floor();
	Vector3i grid_max = (this->aabb.get_end() / cell_size).floor();
	Vector3i grid_size = grid_max + Vector3i(1, 1, 1) - grid_min;

	// uint64_t start = Time::get_singleton()->get_ticks_usec();
	auto oversamples = std::vector<Sample>(oversample_size);
	struct TaskData {
		uint64_t seed;
		float total_area;
		float cell_size;
		Vector<float> *cumulative_areas;
		Vector<Face3> *faces;
		std::vector<Sample> *samples;
		Vector3i grid_min;
		Vector3i grid_size;
	};
	TaskData task_data = {
		seed,
		total_area,
		cell_size,
		&this->cumulative_areas,
		&this->faces,
		&oversamples,
		grid_min,
		grid_size,
	};
	auto task_id = WorkerThreadPool::get_singleton()->add_native_group_task([](void *userdata, uint32_t idx) {
		auto t = *((TaskData *)userdata);
		RandomPCG rng = RandomPCG(t.seed, idx * 3);
		float r = rng.randf() * t.total_area;
		int j = t.cumulative_areas->bsearch(r, true);
		Face3 face = (*t.faces)[j];
		Vector3 a = face.vertex[0];
		Vector3 b = face.vertex[1];
		Vector3 c = face.vertex[2];
		float r1 = Math::sqrt(rng.randf());
		float r2 = rng.randf();
		// https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
		Vector3 pos = (1.0f - r1) * a + r1 * (1.0f - r2) * b + r1 * r2 * c;
		Vector3i cell = (pos / t.cell_size).floor() - t.grid_min;
		uint32_t cell_id = id_from_cell(cell, t.grid_size);
		(*t.samples)[idx] = Sample{ pos, cell_id };
	},
			&task_data, oversample_size);
	WorkerThreadPool::get_singleton()->wait_for_group_task_completion(task_id);
	// print_line("PointSampler::poisson|random ", (Time::get_singleton()->get_ticks_usec() - start) / 1000.0f, "ms");

	// start = Time::get_singleton()->get_ticks_usec();
	std::sort(std::execution::par_unseq, oversamples.begin(), oversamples.end(), [](const Sample &l, const Sample &r) {
		return l.cell_id < r.cell_id;
	});
	// print_line("PointSampler::poisson|oversample_cell_sort ", (Time::get_singleton()->get_ticks_usec() - start) / 1000.0f, "ms");

	// start = Time::get_singleton()->get_ticks_usec();
	auto ggrid = HashMap<uint32_t, Cell>();
	ggrid.reserve(expected_samples * 1.1f);
	auto phases = std::vector<PhaseCell>();
	phases.reserve(expected_samples * 1.1f);
	uint32_t phase_counts[3 * 3 * 3] = {};
	for (uint32_t i = 0; i < oversample_size; i++) {
		Sample s = oversamples[i];
		ggrid.insert(s.cell_id, Cell{ s.cell_id, i, Vector3(FLT_MAX, FLT_MAX, FLT_MAX) });
		Vector3i cell = cell_from_id(s.cell_id, grid_size);
		uint32_t phase_id = (cell.x % 3) + (cell.y % 3) * 3 + (cell.z % 3) * 3 * 3;
		phases.push_back(PhaseCell{ phase_id, s.cell_id });
		phase_counts[phase_id] += 1;
		for (; i + 1 < oversample_size && oversamples[i + 1].cell_id == s.cell_id;) {
			i += 1;
		}
	}
	// print_line("PointSampler::poisson|hash_n_phase ", (Time::get_singleton()->get_ticks_usec() - start) / 1000.0f, "ms");

	// start = Time::get_singleton()->get_ticks_usec();
	std::sort(std::execution::par_unseq, phases.begin(), phases.end(), [](const PhaseCell &l, const PhaseCell &r) {
		return l.phase_id < r.phase_id;
	});
	// print_line("PointSampler::poisson|phases_sort ", (Time::get_singleton()->get_ticks_usec() - start) / 1000.0f, "ms");

	// start = Time::get_singleton()->get_ticks_usec();
	uint32_t phase_count = 0;
	for (int i = 0; i < 3 * 3 * 3; i++) {
		struct TaskData {
			uint32_t phase_count;
			std::vector<PhaseCell> *phases;
			HashMap<uint32_t, Cell> *grid;
			std::vector<Sample> *oversamples;
			float radius;
			uint32_t max_density;
			Vector3i grid_size;
		};
		TaskData task_data = {
			phase_count,
			&phases,
			&ggrid,
			&oversamples,
			radius,
			(uint32_t)Math::ceil(density),
			grid_size,
		};
		auto task_id = WorkerThreadPool::get_singleton()->add_native_group_task([](void *userdata, uint32_t idx) {
			auto t = *((TaskData *)userdata);
			uint32_t phase_idx = t.phase_count + idx;
			auto key = (*t.phases)[phase_idx].cell_id;
			Cell *cell_ptr = t.grid->getptr(key);
			Cell cell = *cell_ptr;
			auto oversamples_size = t.oversamples->size();
			for (uint32_t i = 0; i < t.max_density; i++) {
				auto si = cell.start_idx + i;
				if (si >= oversamples_size || (*t.oversamples)[si].cell_id != cell.cell_id) {
					break;
				}
				auto point = (*t.oversamples)[si].pos;
				for (int xx = -1; xx <= 1; xx++) {
					for (int yy = -1; yy <= 1; yy++) {
						for (int zz = -1; zz <= 1; zz++) {
							if (!(xx == 0 && yy == 0 && zz == 0)) {
								uint32_t nb_key = id_from_cell(cell_from_id(key, t.grid_size) + Vector3(xx, yy, zz), t.grid_size);
								auto it = t.grid->find(nb_key);
								if (it != t.grid->end()) {
									Vector3 nb = it->value.sample;
									if (nb != Vector3(FLT_MAX, FLT_MAX, FLT_MAX) && point.distance_squared_to(nb) < t.radius * t.radius) {
										goto next_point;
									}
								}
							}
						}
					}
				}
				cell_ptr->sample = point;
				break;
			next_point:;
			}
		},
				&task_data, phase_counts[i]);
		WorkerThreadPool::get_singleton()->wait_for_group_task_completion(task_id);
		phase_count += phase_counts[i];
	}
	// print_line("PointSampler::poisson|sample_selection ", (Time::get_singleton()->get_ticks_usec() - start) / 1000.0f, "ms");

	// start = Time::get_singleton()->get_ticks_usec();
	Vector<Vector3> samples;
	samples.resize(expected_samples * 1.1f);
	{
		int i = 0;
		for (const auto &kv : ggrid) {
			if (kv.value.sample != Vector3(FLT_MAX, FLT_MAX, FLT_MAX)) {
				samples.set(i, kv.value.sample);
				i += 1;
			}
		}
		samples.resize(i);
	}
	// print_line("PointSampler::poisson|sample_to_vector ", (Time::get_singleton()->get_ticks_usec() - start) / 1000.0f, "ms");

	return samples;
}

void PointSampler::_bind_methods() {
	ClassDB::bind_method(D_METHOD("init", "mesh"), &PointSampler::init);
	ClassDB::bind_method(D_METHOD("random", "amount", "seed"), &PointSampler::random);
	ClassDB::bind_method(D_METHOD("poisson", "radius", "density", "seed"), &PointSampler::poisson);
}

PointSampler::PointSampler() {
	this->faces = Vector<Face3>();
	this->cumulative_areas = Vector<float>();
}