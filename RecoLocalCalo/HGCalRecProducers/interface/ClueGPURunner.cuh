#include "RecoLocalCalo/HGCalRecProducers/interface/CellsDefinition.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTiles.h"
#include <cuda_runtime.h>
#include <cuda.h>


namespace HGCalRecAlgos{

class ClueGPURunner{
    public:
        CellsOnLayerPtr dc, hc;
        unsigned int cellnum = 1000000;
    
        ClueGPURunner(){
            // std::cout <<"--------CONSTRUCTOR--------" << std::endl;
            init_device();
           
        }
        ~ClueGPURunner(){
            free_device();
        }

        void assign_cells_number(unsigned int numberOfCells){
            cellnum = numberOfCells;
        }

        void init_host(CellsOnLayer& cellsOnLayer){
            hc.x = cellsOnLayer.x.data();
            hc.y = cellsOnLayer.y.data();
            hc.layer = cellsOnLayer.layer.data();
            hc.weight = cellsOnLayer.weight.data();
            hc.sigmaNoise = cellsOnLayer.sigmaNoise.data();
            hc.rho = cellsOnLayer.rho.data();
            hc.delta = cellsOnLayer.delta.data();
            hc.nearestHigher = cellsOnLayer.nearestHigher.data();
            hc.clusterIndex = cellsOnLayer.clusterIndex.data();  
            hc.isSeed = cellsOnLayer.isSeed.data(); 
            // std::cout <<"--------CELLS:"<< cellnum <<" --------" << std::endl;
        }

        void init_device(){
            // std::cout <<"--------Allocating "<< cellnum <<" Cells--------" << std::endl;
            cudaMalloc(&dc.x, sizeof(float)*cellnum);
            cudaMalloc(&dc.y, sizeof(float)*cellnum);
            cudaMalloc(&dc.layer, sizeof(int)*cellnum);
            cudaMalloc(&dc.weight, sizeof(float)*cellnum);
            cudaMalloc(&dc.sigmaNoise, sizeof(float)*cellnum);
            cudaMalloc(&dc.rho, sizeof(float)*cellnum);
            cudaMalloc(&dc.delta, sizeof(float)*cellnum);
            cudaMalloc(&dc.nearestHigher, sizeof(int)*cellnum);
            cudaMalloc(&dc.clusterIndex, sizeof(int)*cellnum);
            cudaMalloc(&dc.isSeed, sizeof(int)*cellnum);
        }

        void copy_todevice(){
            // std::cout <<"Copy to device-------->" << std::endl;
            cudaMemcpy(dc.x, hc.x, sizeof(float)*cellnum, cudaMemcpyHostToDevice);
            cudaMemcpy(dc.y, hc.y, sizeof(float)*cellnum, cudaMemcpyHostToDevice);
            cudaMemcpy(dc.layer, hc.layer, sizeof(int)*cellnum, cudaMemcpyHostToDevice);
            cudaMemcpy(dc.weight, hc.weight, sizeof(float)*cellnum, cudaMemcpyHostToDevice);
            cudaMemcpy(dc.sigmaNoise,hc.sigmaNoise, sizeof(float)*cellnum, cudaMemcpyHostToDevice); 
            cudaMemcpy(dc.isSeed, hc.isSeed, sizeof(int)*cellnum, cudaMemcpyHostToDevice); 
        }

        void clear_set(){
            // std::cout <<"--------Clear--------" << std::endl;
            cudaMemset(dc.rho, 0x00, sizeof(float)*cellnum);
            cudaMemset(dc.delta, 0x00, sizeof(float)*cellnum);
            cudaMemset(dc.nearestHigher, 0x00, sizeof(int)*cellnum);
            cudaMemset(dc.clusterIndex, 0x00, sizeof(int)*cellnum);
            cudaMemset(dc.isSeed, 0x00, sizeof(int)*cellnum);
        }

        void copy_tohost(){
            // std::cout <<"<--------Copy back to host" << std::endl;
            cudaMemcpy(hc.rho, dc.rho, sizeof(float)*cellnum, cudaMemcpyDeviceToHost);
            cudaMemcpy(hc.delta, dc.delta, sizeof(float)*cellnum, cudaMemcpyDeviceToHost);
            cudaMemcpy(hc.nearestHigher, dc.nearestHigher, sizeof(int)*cellnum, cudaMemcpyDeviceToHost);
            cudaMemcpy(hc.clusterIndex, dc.clusterIndex, sizeof(int)*cellnum, cudaMemcpyDeviceToHost);
            cudaMemcpy(hc.isSeed, dc.isSeed, sizeof(int)*cellnum, cudaMemcpyDeviceToHost);
            // std::cout <<"<--------Copy back to host Success" << std::endl;
        }

        void free_device(){
            cudaFree(dc.x);
            cudaFree(dc.y);
            cudaFree(dc.layer);
            cudaFree(dc.weight);
            cudaFree(dc.sigmaNoise);
            cudaFree(dc.rho);
            cudaFree(dc.delta);
            cudaFree(dc.nearestHigher);
            cudaFree(dc.clusterIndex);
            cudaFree(dc.isSeed);
            // std::cout <<"--------FREEDOM---------" << std::endl;
        }

        void clueGPU(std::vector<CellsOnLayer> &, std::vector<int> &, float, float, float, float, float);
};

}
