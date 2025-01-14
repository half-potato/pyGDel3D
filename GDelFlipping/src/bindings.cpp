#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include "gDel3D/GpuDelaunay.h"
#include "DelaunayChecker.h"
#include "InputCreator.h"
#include "gDel3D/GPU/CudaWrapper.h"

void summarize( int pointNum, const GDelOutput& output ) 
{
    ////
    // Summarize on screen
    ////
    std::cout << std::endl; 
    std::cout << "---- SUMMARY ----" << std::endl; 
    std::cout << std::endl; 

    std::cout << "PointNum       " << pointNum                                            << std::endl;
    std::cout << "FP Mode        " << (( sizeof( RealType ) == 8) ? "Double" : "Single")  << std::endl; 
    std::cout << std::endl; 
    std::cout << std::fixed << std::right << std::setprecision( 2 ); 
    std::cout << "TotalTime (ms) " << std::setw( 10 ) << output.stats.totalTime    << std::endl; 
    std::cout << "InitTime       " << std::setw( 10 ) << output.stats.initTime     << std::endl; 
    std::cout << "SplitTime      " << std::setw( 10 ) << output.stats.splitTime    << std::endl; 
    std::cout << "FlipTime       " << std::setw( 10 ) << output.stats.flipTime     << std::endl; 
    std::cout << "RelocateTime   " << std::setw( 10 ) << output.stats.relocateTime << std::endl; 
    std::cout << "SortTime       " << std::setw( 10 ) << output.stats.sortTime     << std::endl;
    std::cout << "OutTime        " << std::setw( 10 ) << output.stats.outTime      << std::endl; 
    std::cout << "SplayingTime   " << std::setw( 10 ) << output.stats.splayingTime << std::endl; 
    std::cout << std::endl;                              
    std::cout << "# Flips        " << std::setw( 10 ) << output.stats.totalFlipNum << std::endl; 
    std::cout << "# Failed verts " << std::setw( 10 ) << output.stats.failVertNum  << std::endl; 
    std::cout << "# Final stars  " << std::setw( 10 ) << output.stats.finalStarNum << std::endl; 
}

class PyGDelOutput {
    public:
        GDelOutput   output; 
        bool checkCorrectness(torch::Tensor &points);
};

bool PyGDelOutput::checkCorrectness(torch::Tensor &points) {
    const Point3* dataPtr = reinterpret_cast<const Point3*>(points.data_ptr());
    
    int64_t size = points.size(0);

    // Create a Thrust host vector
    Point3HVec pointVec(dataPtr, dataPtr + size);
    DelaunayChecker checker( &pointVec,  &output ); 
    checker.checkEuler();
    checker.checkAdjacency();
    checker.checkOrientation();
    return checker.checkDelaunay( false );
}

class PyGPUDel {
  public:
    GpuDel triangulator; 

    PyGPUDel(int N) : triangulator(GDelParams(false, false, false, false)) {
        CudaSafeCall( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );
        triangulator.allocateForFlip(N + 1); // point at infinity
    }
    std::tuple<torch::Tensor, PyGDelOutput> compute(torch::Tensor &points);
    std::tuple<torch::Tensor, PyGDelOutput> computeUsingPrev(torch::Tensor &points, PyGDelOutput &prevOutput);
    ~PyGPUDel() {
        triangulator.cleanup();
    }
};

std::tuple<torch::Tensor, PyGDelOutput>
PyGPUDel::computeUsingPrev(torch::Tensor &points, PyGDelOutput &prevOutput) {
    points = points.cpu().contiguous();
    assert(points.dim() == 2 && points.size(1) == 3);
    // Then match points type with RealType
    if (sizeof(RealType) == 8 && points.scalar_type() != torch::kFloat64) {
        throw std::runtime_error("points must be float64 when RealType is double");
    }
    if (sizeof(RealType) == 4 && points.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("points must be float32 when RealType is float");
    }

    // Access the underlying data pointer and size
    const Point3* dataPtr = reinterpret_cast<const Point3*>(points.data_ptr());
    
    int64_t size = points.size(0);

    // Create a Thrust host vector
    Point3HVec pointVec(dataPtr, dataPtr + size);

    PerfTimer timer; 

    PyGDelOutput pyOutput;
    timer.start(); 
    triangulator.startTiming(); 
    triangulator._output = &pyOutput.output;
    triangulator.initForFlip( pointVec, &prevOutput.output.tetVec );
    // Restore variables
    triangulator._oppVec.copyFromHost(prevOutput.output.tetOppVec);
    triangulator._tetInfoVec.copyFromHost(prevOutput.output.tetInfoVec);


    // triangulator.splitAndFlip();
    triangulator.doFlippingLoop( SphereFastOrientFast ); 

    triangulator.markSpecialTets(); 
    triangulator.doFlippingLoop( SphereExactOrientSoS ); 

    triangulator.relocateAll(); 
    triangulator.outputToHost(); 
    triangulator._splaying.fixWithStarSplaying( pointVec, &pyOutput.output );
    timer.stop(); 
    triangulator._output->stats.totalTime = timer.value(); 

    torch::Tensor output_tensor = torch::from_blob(
        reinterpret_cast<int*>(pyOutput.output.tetVec.data()), 
        {static_cast<long>(pyOutput.output.tetVec.size()), 4},
        torch::TensorOptions().dtype(torch::kInt32)
    );
    return std::make_tuple(output_tensor.clone(), pyOutput);
}

std::tuple<torch::Tensor, PyGDelOutput>
PyGPUDel::compute(torch::Tensor &points) {
    points = points.cpu().contiguous();
    assert(points.dim() == 2 && points.size(1) == 3);
    // Then match points type with RealType
    if (sizeof(RealType) == 8 && points.scalar_type() != torch::kFloat64) {
        throw std::runtime_error("points must be float64 when RealType is double");
    }
    if (sizeof(RealType) == 4 && points.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("points must be float32 when RealType is float");
    }

    // Access the underlying data pointer and size
    const Point3* dataPtr = reinterpret_cast<const Point3*>(points.data_ptr());
    
    int64_t size = points.size(0);

    // Create a Thrust host vector
    Point3HVec pointVec(dataPtr, dataPtr + size);
    PyGDelOutput pyOutput;

    // triangulator.compute( pointVec, &pyOutput.output );

    PerfTimer timer; 
    timer.start(); 
    triangulator.startTiming(); 
    triangulator._output = &pyOutput.output;
    triangulator.initForFlip( pointVec );
    triangulator.splitAndFlip();
    triangulator.outputToHost(); 
    triangulator._splaying.fixWithStarSplaying( pointVec, &pyOutput.output );
    timer.stop(); 
    triangulator._output->stats.totalTime = timer.value(); 

    // summarize( size, output ); 
    torch::Tensor output_tensor = torch::from_blob(
        reinterpret_cast<int*>(pyOutput.output.tetVec.data()), 
        {static_cast<long>(pyOutput.output.tetVec.size()), 4},
        torch::TensorOptions().dtype(torch::kInt32)
    );
    return std::make_tuple(output_tensor.clone(), pyOutput);
}


namespace py = pybind11;

PYBIND11_MODULE(gdel3d, m) {
    m.doc() = "Python bindings for gdel3d";
    py::class_<PyGDelOutput>(m, "DelOutput")
        .def("check_correctness", &PyGDelOutput::checkCorrectness);
    py::class_<PyGPUDel>(m, "Del")
        .def(py::init<int>())
        .def("compute", &PyGPUDel::compute)
        .def("compute_from_prev", &PyGPUDel::computeUsingPrev);
}
