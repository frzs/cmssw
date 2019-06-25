struct CellsOnLayerPtr
{
  float *x; 
  float *y ;
  int *layer ;
  float *weight ;
  float *sigmaNoise; 

  float *rho ; 
  float *delta; 
  int *nearestHigher;
  int *clusterIndex; 
  int *isSeed;
};

  // void initHost(CellsOnLayer<float>& cellsOnLayer ){
  //   x = cellsOnLayer.x.data();
  //   y = cellsOnLayer.y.data();
  //   layer = cellsOnLayer.layer.data();
  //   weight = cellsOnLayer.weight.data();
  //   sigmaNoise = cellsOnLayer.sigmaNoise.data();

  //   rho = cellsOnLayer.rho.data();
  //   delta = cellsOnLayer.delta.data();
  //   nearestHigher = cellsOnLayer.nearestHigher.data();
  //   clusterIndex = cellsOnLayer.clusterIndex.data();
  //   isSeed = cellsOnLayer.isSeed.data();
  // }

  // void initDevice(CellsOnLayerPtr h_cells, unsigned int numberOfCells){
  //   cudaMalloc(&x, sizeof(float)*numberOfCells);
  //   cudaMalloc(&y, sizeof(float)*numberOfCells);
  //   cudaMalloc(&layer, sizeof(int)*numberOfCells);
  //   cudaMalloc(&weight, sizeof(float)*numberOfCells);
  //   cudaMalloc(&sigmaNoise, sizeof(float)*numberOfCells);
  //   cudaMemcpy(x, h_cells.x, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
  //   cudaMemcpy(y, h_cells.y, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
  //   cudaMemcpy(layer, h_cells.layer, sizeof(int)*numberOfCells, cudaMemcpyHostToDevice);
  //   cudaMemcpy(weight, h_cells.weight, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
  //   cudaMemcpy(sigmaNoise, h_cells.sigmaNoise, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice); 


  //   cudaMalloc(&rho, sizeof(float)*numberOfCells);
  //   cudaMemset(rho, 0x00, sizeof(float)*numberOfCells);
  //   cudaMalloc(&delta, sizeof(float)*numberOfCells);
  //   cudaMemset(delta, 0x00, sizeof(float)*numberOfCells);
  //   cudaMalloc(&nearestHigher, sizeof(int)*numberOfCells);
  //   cudaMemset(nearestHigher, 0x00, sizeof(int)*numberOfCells);
  //   cudaMalloc(&clusterIndex, sizeof(int)*numberOfCells);
  //   cudaMemset(clusterIndex, 0x00, sizeof(int)*numberOfCells);
  //   cudaMalloc(&isSeed, sizeof(int)*numberOfCells);
  //   cudaMemset(isSeed, 0x00, sizeof(int)*numberOfCells);
  //   cudaMemcpy(isSeed, h_cells.isSeed, sizeof(int)*numberOfCells, cudaMemcpyHostToDevice); 
  // }

  // // void allocMem(unsigned int numberOfCells){
  // //   cudaMalloc(&x, sizeof(float)*numberOfCells);
  // //   cudaMalloc(&y, sizeof(float)*numberOfCells);
  // //   cudaMalloc(&layer, sizeof(int)*numberOfCells);
  // //   cudaMalloc(&weight, sizeof(float)*numberOfCells);
  // //   cudaMalloc(&sigmaNoise, sizeof(float)*numberOfCells);
    
  // //   cudaMalloc(&rho, sizeof(float)*numberOfCells);
  // //   cudaMemset(rho, 0x00, sizeof(float)*numberOfCells);
  // //   cudaMalloc(&delta, sizeof(float)*numberOfCells);
  // //   cudaMemset(delta, 0x00, sizeof(float)*numberOfCells);
  // //   cudaMalloc(&nearestHigher, sizeof(int)*numberOfCells);
  // //   cudaMemset(nearestHigher, 0x00, sizeof(int)*numberOfCells);
  // //   cudaMalloc(&clusterIndex, sizeof(int)*numberOfCells);
  // //   cudaMemset(clusterIndex, 0x00, sizeof(int)*numberOfCells);
  // //   cudaMalloc(&isSeed, sizeof(int)*numberOfCells);
  // //   cudaMemset(isSeed, 0x00, sizeof(int)*numberOfCells);
  // // }

  // void cpyDToH(CellsOnLayerPtr h_cells, unsigned int numberOfCells){
  //   cudaMemcpy(h_cells.rho, rho, sizeof(float)*numberOfCells, cudaMemcpyDeviceToHost);
  //   cudaMemcpy(h_cells.delta, delta, sizeof(float)*numberOfCells, cudaMemcpyDeviceToHost);
  //   cudaMemcpy(h_cells.nearestHigher, nearestHigher, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
  //   cudaMemcpy(h_cells.clusterIndex, clusterIndex, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
  //   cudaMemcpy(h_cells.isSeed, isSeed, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
  // }

    


  // void freeDevice(){
  //   cudaFree(x);
  //   cudaFree(y);
  //   cudaFree(layer);
  //   cudaFree(weight);
  //   cudaFree(sigmaNoise);
    

  //   cudaFree(rho);
  //   cudaFree(delta);
  //   cudaFree(nearestHigher);
  //   cudaFree(clusterIndex);
  //   cudaFree(isSeed);
    
  // }

// };

