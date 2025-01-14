#define CGAL_NO_POSTCONDITIONS 1
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <vector>
#include <unordered_map>
// #include <torch/extension.h>

namespace py = pybind11;

using namespace CGAL;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<std::size_t, K> Vb;
typedef CGAL::Triangulation_data_structure_3<Vb> Tds;
  // typedef CGAL::Triangulation_data_structure_3<
  //           // CGAL::Triangulation_vertex_base_3<K>,
  //           Vb,
  //           CGAL::Delaunay_triangulation_cell_base_3<K>,
  //           CGAL::Parallel_tag>                               Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> Delaunay;
typedef Delaunay::Point Point;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Cell_handle Cell_handle;
typedef std::pair<Point, Point> Circles;
typedef Vector_3<K> Vec;

static const float TINY_VAL = 1.0754944e-20;
static const float MIN_VAL = -1e+20;
static const float MAX_VAL = 1e+20;

// float clip(float v, float minv, float maxv) {
//   return fmax(fmin(v, maxv), minv);
// }
//
// float safe_div(float a, float b) {
//     if (abs(b) < TINY_VAL) {
//         return clip(a / TINY_VAL, MIN_VAL, MAX_VAL);
//     } else {
//         return clip(a / b, MIN_VAL, MAX_VAL);
//     }
// }
//
// Vec safe_div(Vec a, float b) {
//     return Vec(
//         safe_div(a.x(), b),
//         safe_div(a.y(), b),
//         safe_div(a.z(), b));
// }
//
// Point calculate_circumcenter(Point v1, Point v2, Point v3, Point v4) {
//     Vec a = v2 - v1;
//     Vec b = v3 - v1;
//     Vec c = v4 - v1;
//
//     float aa = scalar_product(a, a);
//     float bb = scalar_product(b, b);
//     float cc = scalar_product(c, c);
//
//     Vec cross_bc = cross_product(b, c);
//     Vec cross_ca = cross_product(c, a);
//     Vec cross_ab = cross_product(a, b);
//
//     float denominator = 2.f * scalar_product(a, cross_bc);
//
//     Vec relative_circumcenter = safe_div(
//         aa * cross_bc +
//         bb * cross_ca +
//         cc * cross_ab
//     , denominator);
//     
//     // Return absolute position
//     return v1 + relative_circumcenter;
// }

// float norm(Vec a) {
//     return sqrt(a.x()*a.x() + a.y()*a.y() + a.z()*a.z());
// }

class DelaunayTriangulation {
private:
    Delaunay triangulation;
    std::size_t next_index;

public:
    DelaunayTriangulation() : next_index(0) {}

    void init_from_points(py::array_t<float> points) {
        auto r = points.unchecked<2>();
        if (r.shape(1) != 3) {
            throw std::runtime_error("Points array must be Nx3");
        }

        // Clear existing triangulation
        triangulation.clear();
        next_index = 0;
        add_points(points);
    }

    void add_points(py::array_t<double> points) {
        auto r = points.unchecked<2>();
        if (r.shape(1) != 3) {
            throw std::runtime_error("Points array must be Nx3");
        }

        std::vector<std::pair<Point, std::size_t>> points_with_info;
        points_with_info.reserve(r.shape(0));

        // Convert numpy array to CGAL points with indices
        float minx = std::numeric_limits<float>::max();
        float miny = std::numeric_limits<float>::max();
        float minz = std::numeric_limits<float>::max();
        float maxx = std::numeric_limits<float>::lowest();
        float maxy = std::numeric_limits<float>::lowest();
        float maxz = std::numeric_limits<float>::lowest();
        for (ssize_t i = 0; i < r.shape(0); ++i) {
            float x = r(i, 0);
            float y = r(i, 1);
            float z = r(i, 2);
            
            minx = std::min(minx, x);
            miny = std::min(miny, y);
            minz = std::min(minz, z);
            maxx = std::max(maxx, x);
            maxy = std::max(maxy, y);
            maxz = std::max(maxz, z);
            Point p(r(i, 0), r(i, 1), r(i, 2));
            points_with_info.emplace_back(p, next_index);
            next_index++;
        }

        // Insert new points
        // Delaunay::Lock_data_structure locking_ds(
        //   CGAL::Bbox_3(minx, miny, minz, maxx, maxy, maxz), 50);
        // triangulation.set_lock_data_structure(&locking_ds);
        // printf("Is parallel: %i, %i\n", triangulation.is_parallel(), CGAL_LINKED_WITH_TBB);
        triangulation.insert(points_with_info.begin(), points_with_info.end());//, &locking_ds);

    }

    void sparse_update_points(
        py::array_t<bool> mask, py::array_t<float> points) {
        auto r = points.unchecked<2>();
        if (r.shape(1) != 3) {
            throw std::runtime_error("Points array must be Nx3");
        }
        if (r.shape(0) != num_vertices()) {
            throw std::runtime_error("Point array dimension 0 must be equal to current number points");
        }

        auto m = mask.unchecked<1>();
        if (m.shape(0) != num_vertices()) {
            throw std::runtime_error("Mask dimension 0 must be equal to current number points");
        }
        // This causes violations of the Delaunay Triangulation

        for (Delaunay::Vertex_handle v : triangulation.finite_vertex_handles()) {
            size_t idx = v->info();
            if (idx >= 0 && idx < r.shape(0)) {
                Point p(r(idx, 0), r(idx, 1), r(idx, 2));
                if (m(idx)) {
                    triangulation.move_if_no_collision(v, p);
                }
            }
        }
        for (Delaunay::Vertex_handle v : triangulation.finite_vertex_handles()) {
            size_t idx = v->info();
            if (idx >= 0 && idx < r.shape(0)) {
                Point p(r(idx, 0), r(idx, 1), r(idx, 2));
                if (!m(idx)) {
                    v->set_point(p);
                }
            }
        }
        // std::vector<std::pair<Point, std::size_t>> points_with_info;
        // for (Delaunay::Vertex_handle v : triangulation.finite_vertex_handles()) {
        //     size_t idx = v->info();
        //     if (idx >= 0 && idx < r.shape(0)) {
        //         Point p(r(idx, 0), r(idx, 1), r(idx, 2));
        //         if (m(idx)) {
        //             triangulation.remove(v);
        //             points_with_info.emplace_back(p, idx);
        //         }
        //     }
        // }
        // triangulation.insert(points_with_info.begin(), points_with_info.end());//, &locking_ds);
    }

    // py::array_t<size_t> smart_update_points(
    //     py::array_t<float> points,
    //     py::array_t<float> new_points)
    // {
    //     auto r = points.unchecked<2>();
    //     auto n = new_points.unchecked<2>();
    //     if (r.shape(1) != 3) {
    //         throw std::runtime_error("Points array must be Nx3");
    //     }
    //     if (r.shape(0) != num_vertices()) {
    //         throw std::runtime_error("Point array dimension 0 must be equal to current number points");
    //     }
    //
    //     // First, develop a list of points we actually need to move
    //     //
    //     // distances, neighbors = tree.query(old_centers, k=2)  # k=2 to get nearest non-self neighbor
    //     // delta_c = np.linalg.norm(new_centers - old_centers, axis=-1)  # movement of this center
    //     // delta_n = np.linalg.norm(new_centers[neighbors[:, 1]] - old_centers[neighbors[:, 1]], axis=-1)  # movement of nearest center
    //     // violations, = np.where(distances[:, 1] < (delta_c + delta_n))
    //     // violation_points = np.unique(indices_np[violations].reshape(-1))
    //
    //     // std::vector<bool> mask(r.shape(0), false);
    //
    //     // Calculate circumcircles
    //     size_t idx = 0;
    //     std::unordered_map<Cell_handle, Circles> circumcenters(num_cells());
    //     for (auto cell = triangulation.cells_begin(); 
    //          cell != triangulation.cells_end(); ++cell) {
    //         bool valid_cell = true;
    //         for (int i = 0; i < 4; ++i) {
    //             if (cell->vertex(i)->info() > triangulation.number_of_vertices()) {
    //                 valid_cell = false;
    //                 break;
    //             }
    //         }
    //         if (!valid_cell) continue;
    //
    //
    //         size_t a = cell->vertex(0)->info();
    //         size_t b = cell->vertex(1)->info();
    //         size_t c = cell->vertex(2)->info();
    //         size_t d = cell->vertex(3)->info();
    //         circumcenters[cell] = std::make_pair(
    //             calculate_circumcenter(
    //                 cell->vertex(0)->point(), cell->vertex(1)->point(),
    //                 cell->vertex(2)->point(), cell->vertex(3)->point()),
    //             calculate_circumcenter(
    //                 Point(n(a, 0), n(a, 1), n(a, 2)),
    //                 Point(n(b, 0), n(b, 1), n(b, 2)),
    //                 Point(n(c, 0), n(c, 1), n(c, 2)),
    //                 Point(n(d, 0), n(d, 1), n(d, 2)))
    //         );
    //         idx++;
    //     }
    //     /*
    //     idx = 0;
    //     for (auto cell = triangulation.cells_begin(); 
    //          cell != triangulation.cells_end(); ++cell) {
    //         bool valid_cell = true;
    //         for (int i = 0; i < 4; ++i) {
    //             if (cell->vertex(i)->info() > triangulation.number_of_vertices()) {
    //                 valid_cell = false;
    //                 break;
    //             }
    //         }
    //         if (!valid_cell) continue;
    //
    //         Point old_center, new_center, n_old_center, n_new_center;
    //         std::tie(old_center, new_center) = circumcenters[cell];
    //         float delta_c = norm(old_center - new_center);
    //         std::vector<Cell_handle> out_cells;
    //
    //         for (int i = 0; i < 4; ++i) {
    //             // The rest is just to iterate over cells
    //             size_t idx = cell->vertex(i)->info();
    //             // reset inserter
    //             out_cells.resize(0);
    //             triangulation.incident_cells(cell->vertex(i), std::back_inserter(out_cells));
    //             for (int j = 0; j<out_cells.size(); j++) {
    //                 Cell_handle &n_cell = out_cells[j];
    //                 if (cell == n_cell) continue;
    //                 // get circumcircle
    //                 std::tie(n_old_center, n_new_center) = circumcenters[n_cell];
    //                 float delta_n = norm(n_old_center - n_new_center);
    //                 if (norm(old_center - n_old_center) < (delta_c + delta_n)) {
    //                     mask[idx] = true;
    //                 }
    //             }
    //         }
    //         idx++;
    //     }
    //     */
    //
    //     std::vector<Cell_handle> out_cells;
    //     std::vector<size_t> removal_list;
    //     Point old_center, new_center, n_old_center, n_new_center;
    //     size_t potential_violations = 0;
    //     for (ssize_t i = 0; i < r.shape(0); ++i) {
    //         Point p(r(i, 0), r(i, 1), r(i, 2));
    //         auto it = index_to_vertex.find(i);
    //         if (it != index_to_vertex.end()) {
    //
    //             out_cells.resize(0);
    //             triangulation.incident_cells(it->second, std::back_inserter(out_cells));
    //             bool potential_violation = false;
    //             // We need to do an all to all comparison of the distances to detect if any adjacent cells could be in violation
    //             for (int j = 0; j<out_cells.size(); j++) {
    //                 Cell_handle &cell = out_cells[j];
    //                 // get circumcircle
    //                 std::tie(old_center, new_center) = circumcenters[cell];
    //                 float delta_c = norm(old_center - new_center);
    //                 for (int k = j+1; k<out_cells.size(); k++) {
    //                     Cell_handle &n_cell = out_cells[k];
    //                     if (cell == n_cell) continue;
    //                     std::tie(n_old_center, n_new_center) = circumcenters[n_cell];
    //                     float delta_n = norm(n_old_center - n_new_center);
    //
    //                     if (norm(old_center - n_old_center) < (delta_c + delta_n)) {
    //                         potential_violation = true;
    //                         break;
    //                     }
    //                 }
    //                 if (potential_violation) break;
    //             }
    //             // if (mask[i]) {
    //             if (potential_violation) {
    //                 potential_violations++;
    //                 Vertex_handle colliding_vertex = triangulation.move(it->second, p);
    //                 // if (colliding_vertex != nullptr) {
    //                 //     removal_list.push_back(i);
    //                 // }
    //             } else {
    //                 it->second->set_point(p);
    //             }
    //         }
    //     }
    //     printf("%i\n", potential_violations);
    //     py::array_t<size_t> arr = py::array_t<size_t>(
    //         {removal_list.size()},  // shape
    //         {sizeof(size_t)},  // stride
    //         removal_list.data()  // data pointer
    //     );
    //     return arr;
    // }
    //

    void update_points(py::array_t<float> points) {
        auto r = points.unchecked<2>();
        if (r.shape(1) != 3) {
            throw std::runtime_error("Points array must be Nx3");
        }
        if (r.shape(0) != num_vertices()) {
            throw std::runtime_error("Number of points must be equal to current number points");
        }

        for (Delaunay::Vertex_handle v : triangulation.finite_vertex_handles()) {
            size_t idx = v->info();
            if (idx >= 0 && idx < r.shape(0)) {
                Point p(r(idx, 0), r(idx, 1), r(idx, 2));
                triangulation.move_if_no_collision(v, p);
            }
        }
    }

    size_t num_vertices() const {
        return triangulation.number_of_vertices();
    }

    size_t num_cells() const {
        return triangulation.number_of_finite_cells();
    }

    py::array_t<size_t> get_cells() const {
        size_t n = triangulation.number_of_finite_cells();
        
        // Create array with the right shape
        py::array_t<size_t> result(py::array::ShapeContainer({static_cast<ssize_t>(n), 4}));
        
        // Get buffer for direct memory access
        auto buf = result.mutable_unchecked<2>();

        size_t idx = 0;
        for (Delaunay::Cell_handle cell : triangulation.finite_cell_handles()) {
            for (int i = 0; i < 4; ++i) {
                buf(idx, i) = cell->vertex(i)->info();
            }
            idx++;
        }

        return result;
    }
};

PYBIND11_MODULE(dtet, m) {
    m.doc() = "Python bindings for 3D Delaunay triangulation using CGAL";
    
    py::class_<DelaunayTriangulation>(m, "DelaunayTriangulation")
        .def(py::init<>())
        .def("init_from_points", &DelaunayTriangulation::init_from_points,
             "Initialize triangulation from Nx3 numpy array of points")
        .def("add_points", &DelaunayTriangulation::add_points,
             "Add points to existing triangulation")
        .def("update_points", &DelaunayTriangulation::update_points,
             "Update all point locations")
        .def("sparse_update_points", &DelaunayTriangulation::sparse_update_points,
             "Sparse update all point locations")
        .def("num_vertices", &DelaunayTriangulation::num_vertices,
             "Get number of vertices")
        .def("num_cells", &DelaunayTriangulation::num_cells,
             "Get number of cells (tetrahedra)")
        .def("get_cells", &DelaunayTriangulation::get_cells,
             "Get Mx4 array of vertex indices for each tetrahedron");
}
