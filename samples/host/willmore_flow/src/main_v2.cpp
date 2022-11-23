#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <numeric>
#include <igl/write_triangle_mesh.h>
#include <igl/read_triangle_mesh.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>

#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/willmore_flow.hpp"
#include "flo/host/surface.hpp"

#include <CGAL/IO/OBJ.h>
#include <CGAL/IO/OBJ/File_writer_wavefront.h>
#include <CGAL/Polyhedron_3.h>

using namespace Eigen;

template <typename T>
struct ForwardEuler
{
  T tao = 0.95f;

  ForwardEuler(T i_tao) : tao(std::move(i_tao))
  {
  }

  void operator()(Eigen::Matrix<T, Eigen::Dynamic, 1>& i_x,
                  const Eigen::Matrix<T, Eigen::Dynamic, 1>& i_dx) const
  {
    i_x += i_dx * tao;
  }
};

int main(int argc, char* argv[])
{
  // Command line arguments
  const std::string in_name = argv[1];
  const std::string out_name = argv[2];
  const int max_iter = std::stoi(argv[3]);
  // const flo::real tao = std::stof(argv[4]);

  flo::host::Surface surf;

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;

  std::cout << "\n\n" << in_name << "\n\n";

  std::vector<CGAL::Simple_cartesian<double>::Point_3> points_ref;
  std::vector<std::vector<std::size_t>> faces_ref;

  std::cout << CGAL::IO::read_OBJ(in_name, points_ref, faces_ref) << "\n";

  // igl::readOFF(in_name, V, F);

  // igl::read_triangle_mesh("obj",in_name, surf.vertices, surf.faces);
  // bool success = igl::readOBJ(in_name, surf.vertices, surf.faces);

  // ForwardEuler<flo::real> integrator(tao);

  // for (int iter = 0; iter < max_iter; ++iter)
  // {
  //   std::cout << "Iteration: " << iter << '\n';
  //   flo::host::willmore_flow(surf.vertices, surf.faces, integrator);
  // }

  // igl::write_triangle_mesh(out_name, surf.vertices, surf.faces);

  return EXIT_SUCCESS;
}
