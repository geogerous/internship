# Code Citations

## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res *
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copyParams.dstArray
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copyParams.dstArray = mips_dev[mip];
        copyParams.extent
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copyParams.dstArray = mips_dev[mip];
        copyParams.extent = make_cudaExtent(source_res, source_res, source_res);
        copyParams
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copyParams.dstArray = mips_dev[mip];
        copyParams.extent = make_cudaExtent(source_res, source_res, source_res);
        copyParams.kind = cudaMemcpyHostToDevice;
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copy
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copyParams.dstArray = m
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copyParams.dstArray = mips_dev[mip];
        copyParams.extent = make_cudaExtent(source_res, source_res, source_res);
        copyParams.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copyParams.dstArray = mips_dev[mip];
        copyParams.extent = make_cudaExtent(source
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copyParams.dstArray = mips_dev[mip];
        copyParams.extent = make_cudaExtent(source_res, source_res, source
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copyParams.dstArray = mips_dev[mip];
        copyParams.extent = make_cudaExtent(source_res, source_res, source_res);
        copyParams.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
mip < 9; mip++) {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copyParams.dstArray = mips_dev[mip];
        copyParams.extent = make_cudaExtent(source_res, source_res, source_res);
        copyParams.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        CheckError;
    }

    #define BindMip(i)  Mip(i).normalized = true;\
                        Mip(i).filterMode = cudaFilterModeLinear;\
                        Mip(i).addressMode[0] = cudaAddressModeBorder;\
                        Mip(i).addressMode[1] = cudaAddressModeBorder;\
                        Mip(i).addressMode[2] = cudaAddressModeBorder;\
                        cudaBindTextureToArray(Mip(i), mips_dev[i], channel_desc);

    BindMip(0); BindMip(1); BindMip(2);
    BindMip(3); BindMip(4
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        CheckError;
    }

    #define BindMip(i)  Mip(i).normalized = true;\
                        Mip(i).filterMode = cudaFilterModeLinear;\
                        Mip(i).addressMode[0] = cudaAddressModeBorder;\
                        Mip(i).addressMode[1] = cudaAddressModeBorder;\
                        Mip(i).addressMode[2] = cudaAddressModeBorder;\
                        cudaBindTextureToArray(Mip(i), mips_dev[i], channel_desc);

    BindMip(0); BindMip(1); BindMip(2);
    BindMip(3); BindMip(4); BindMip(5);
    BindMip
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        CheckError;
    }

    #define BindMip(i)  Mip(i).normalized = true;\
                        Mip(i).filterMode = cudaFilterModeLinear;\
                        Mip(i).addressMode[0] = cudaAddressModeBorder;\
                        Mip(i).addressMode[1] = cudaAddressModeBorder;\
                        Mip(i).addressMode[2] = cudaAddressModeBorder;\
                        cudaBindTextureToArray(Mip(i), mips_dev[i], channel_desc);

    BindMip(0); BindMip(1); BindMip(2);
    BindMip(3); BindMip(4); BindMip(5);
    BindMip(6); BindMip(7); BindMip(8
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        CheckError;
    }

    #define BindMip(i)  Mip(i).normalized = true;\
                        Mip(i).filterMode = cudaFilterModeLinear;\
                        Mip(i).addressMode[0] = cudaAddressModeBorder;\
                        Mip(i).addressMode[1] = cudaAddressModeBorder;\
                        Mip(i).addressMode[2] = cudaAddressModeBorder;\
                        cudaBindTextureToArray(Mip(i), mips_dev[i], channel_desc);

    BindMip(0); BindMip(1); BindMip(2);
    BindMip(3); BindMip(4); BindMip(5);
    BindMip(6); BindMip(7); BindMip(8
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        CheckError;
    }

    #define BindMip(i)  Mip(i).normalized = true;\
                        Mip(i).filterMode = cudaFilterModeLinear;\
                        Mip(i).addressMode[0] = cudaAddressModeBorder;\
                        Mip(i).addressMode[1] = cudaAddressModeBorder;\
                        Mip(i).addressMode[2] = cudaAddressModeBorder;\
                        cudaBindTextureToArray(Mip(i), mips_dev[i], channel_desc);

    BindMip(0); BindMip(1); BindMip(2);
    BindMip(3); BindMip(4); BindMip(5);
    BindMip(6); BindMip(7); BindMip(8
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        CheckError;
    }

    #define BindMip(i)  Mip(i).normalized = true;\
                        Mip(i).filterMode = cudaFilterModeLinear;\
                        Mip(i).addressMode[0] = cudaAddressModeBorder;\
                        Mip(i).addressMode[1] = cudaAddressModeBorder;\
                        Mip(i).addressMode[2] = cudaAddressModeBorder;\
                        cudaBindTextureToArray(Mip(i), mips_dev[i], channel_desc);

    BindMip(0); BindMip(1); BindMip(2);
    BindMip(3); BindMip(4); BindMip(5);
    BindMip(6); BindMip(7); BindMip(8
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        CheckError;
    }

    #define BindMip(i)  Mip(i).normalized = true;\
                        Mip(i).filterMode = cudaFilterModeLinear;\
                        Mip(i).addressMode[0] = cudaAddressModeBorder;\
                        Mip(i).addressMode[1] = cudaAddressModeBorder;\
                        Mip(i).addressMode[2] = cudaAddressModeBorder;\
                        cudaBindTextureToArray(Mip(i), mips_dev[i], channel_desc);

    BindMip(0); BindMip(1); BindMip(2);
    BindMip(3); BindMip(4); BindMip(5);
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        CheckError;
    }

    #define BindMip(i)  Mip(i).normalized = true;\
                        Mip(i).filterMode = cudaFilterModeLinear;\
                        Mip(i).addressMode[0] = cudaAddressModeBorder;\
                        Mip(i).addressMode[1] = cudaAddressModeBorder;\
                        Mip(i).addressMode[2] = cudaAddressModeBorder;\
                        cudaBindTextureToArray(Mip(i), mips_dev[i], channel_desc);

    BindMip(0); BindMip(1); BindMip(2);
    BindMip(3); BindMip(4); BindMip(5);
    BindMip(6);
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        CheckError;
    }

    #define BindMip(i)  Mip(i).normalized = true;\
                        Mip(i).filterMode = cudaFilterModeLinear;\
                        Mip(i).addressMode[0] = cudaAddressModeBorder;\
                        Mip(i).addressMode[1] = cudaAddressModeBorder;\
                        Mip(i).addressMode[2] = cudaAddressModeBorder;\
                        cudaBindTextureToArray(Mip(i), mips_dev[i], channel_desc);

    BindMip(0); BindMip(1); BindMip(2);
    BindMip(3); BindMip(4); BindMip(5);
    BindMip(6); BindMip(7); 
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/volume.cu

```
cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        CheckError;
    }

    #define BindMip(i)  Mip(i).normalized = true;\
                        Mip(i).filterMode = cudaFilterModeLinear;\
                        Mip(i).addressMode[0] = cudaAddressModeBorder;\
                        Mip(i).addressMode[1] = cudaAddressModeBorder;\
                        Mip(i).addressMode[2] = cudaAddressModeBorder;\
                        cudaBindTextureToArray(Mip(i), mips_dev[i], channel_desc);

    BindMip(0); BindMip(1); BindMip(2);
    BindMip(3); BindMip(4); BindMip(5);
    BindMip(6); BindMip(7); BindMip(8
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos =
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            Current
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer +=
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(ran
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex,
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * Current
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
= normalize(CurrentOffsetPos - pos);
                float Radius = float(1 << int(CurrentOffsetInfo.Layer)) / 256.0f;
                float Angle = atan(0.5f * Radius / CurrentOffsetInfo.Offset);
                float cos = dot(XMain, MsDir);
                float cos2 = dot(LXMain, MsDir);
                float u = cos * 0.5f + 0.5f;
                float u2 = cos2 * 0.5f + 0.5f;
                float v = Angle / (3.1415926535f * 60.0f / 180.0f);
                float HG0 = tex2D<float>(_HGLut, u, v)
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
= normalize(CurrentOffsetPos - pos);
                float Radius = float(1 << int(CurrentOffsetInfo.Layer)) / 256.0f;
                float Angle = atan(0.5f * Radius / CurrentOffsetInfo.Offset);
                float cos = dot(XMain, MsDir);
                float cos2 = dot(LXMain, MsDir);
                float u = cos * 0.5f + 0.5f;
                float u2 = cos2 * 0.5f + 0.5f;
                float v = Angle / (3.1415926535f * 60.0f / 180.0f);
                float HG0 = tex2D<float>(_HGLut, u, v);
                float HG1 = tex2D<
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
= normalize(CurrentOffsetPos - pos);
                float Radius = float(1 << int(CurrentOffsetInfo.Layer)) / 256.0f;
                float Angle = atan(0.5f * Radius / CurrentOffsetInfo.Offset);
                float cos = dot(XMain, MsDir);
                float cos2 = dot(LXMain, MsDir);
                float u = cos * 0.5f + 0.5f;
                float u2 = cos2 * 0.5f + 0.5f;
                float v = Angle / (3.1415926535f * 60.0f / 180.0f);
                float HG0 = tex2D<float>(_HGLut, u, v);
                float HG1 = tex2D<float>(_HGLut,
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
= normalize(CurrentOffsetPos - pos);
                float Radius = float(1 << int(CurrentOffsetInfo.Layer)) / 256.0f;
                float Angle = atan(0.5f * Radius / CurrentOffsetInfo.Offset);
                float cos = dot(XMain, MsDir);
                float cos2 = dot(LXMain, MsDir);
                float u = cos * 0.5f + 0.5f;
                float u2 = cos2 * 0.5f + 0.5f;
                float v = Angle / (3.1415926535f * 60.0f / 180.0f);
                float HG0 = tex2D<float>(_HGLut, u, v);
                float HG1 = tex2D<float>(_HGLut, u2, v);
                X_Val_Hg[i]
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
= normalize(CurrentOffsetPos - pos);
                float Radius = float(1 << int(CurrentOffsetInfo.Layer)) / 256.0f;
                float Angle = atan(0.5f * Radius / CurrentOffsetInfo.Offset);
                float cos = dot(XMain, MsDir);
                float cos2 = dot(LXMain, MsDir);
                float u = cos * 0.5f + 0.5f;
                float u2 = cos2 * 0.5f + 0.5f;
                float v = Angle / (3.1415926535f * 60.0f / 180.0f);
                float HG0 = tex2D<float>(_HGLut, u, v);
                float HG1 = tex2D<float>(_HGLut, u2, v);
                X_Val_Hg[i] = log(HG0 *
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
= normalize(CurrentOffsetPos - pos);
                float Radius = float(1 << int(CurrentOffsetInfo.Layer)) / 256.0f;
                float Angle = atan(0.5f * Radius / CurrentOffsetInfo.Offset);
                float cos = dot(XMain, MsDir);
                float cos2 = dot(LXMain, MsDir);
                float u = cos * 0.5f + 0.5f;
                float u2 = cos2 * 0.5f + 0.5f;
                float v = Angle / (3.1415926535f * 60.0f / 180.0f);
                float HG0 = tex2D<float>(_HGLut, u, v);
                float HG1 = tex2D<float>(_HGLut, u2, v);
                X_Val_Hg[i] = log(HG0 * HG1 + 1.0f);
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
= normalize(CurrentOffsetPos - pos);
                float Radius = float(1 << int(CurrentOffsetInfo.Layer)) / 256.0f;
                float Angle = atan(0.5f * Radius / CurrentOffsetInfo.Offset);
                float cos = dot(XMain, MsDir);
                float cos2 = dot(LXMain, MsDir);
                float u = cos * 0.5f + 0.5f;
                float u2 = cos2 * 0.5f + 0.5f;
                float v = Angle / (3.1415926535f * 60.0f / 180.0f);
                float HG0 = tex2D<float>(_HGLut, u, v);
                float HG1 = tex2D<float>(_HGLut, u2, v);
                X_Val_Hg[i] = log(HG0 * HG1 + 1.0f);
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
= normalize(CurrentOffsetPos - pos);
                float Radius = float(1 << int(CurrentOffsetInfo.Layer)) / 256.0f;
                float Angle = atan(0.5f * Radius / CurrentOffsetInfo.Offset);
                float cos = dot(XMain, MsDir);
                float cos2 = dot(LXMain, MsDir);
                float u = cos * 0.5f + 0.5f;
                float u2 = cos2 * 0.5f + 0.5f;
                float v = Angle / (3.1415926535f * 60.0f / 180.0f);
                float HG0 = tex2D<float>(_HGLut, u, v);
                float HG1 = tex2D<float>(_HGLut, u2, v);
                X_Val_Hg[i] = log(HG0 * HG1 + 1.0f);
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
= normalize(CurrentOffsetPos - pos);
                float Radius = float(1 << int(CurrentOffsetInfo.Layer)) / 256.0f;
                float Angle = atan(0.5f * Radius / CurrentOffsetInfo.Offset);
                float cos = dot(XMain, MsDir);
                float cos2 = dot(LXMain, MsDir);
                float u = cos * 0.5f + 0.5f;
                float u2 = cos2 * 0.5f + 0.5f;
                float v = Angle / (3.1415926535f * 60.0f / 180.0f);
                float HG0 = tex2D<float>(_HGLut, u, v);
                float HG1 = tex2D<float>(_HGLut, u2, v);
                X_Val_Hg[i] = log(HG0 * HG1 + 1.0f);
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
= normalize(CurrentOffsetPos - pos);
                float Radius = float(1 << int(CurrentOffsetInfo.Layer)) / 256.0f;
                float Angle = atan(0.5f * Radius / CurrentOffsetInfo.Offset);
                float cos = dot(XMain, MsDir);
                float cos2 = dot(LXMain, MsDir);
                float u = cos * 0.5f + 0.5f;
                float u2 = cos2 * 0.5f + 0.5f;
                float v = Angle / (3.1415926535f * 60.0f / 180.0f);
                float HG0 = tex2D<float>(_HGLut, u, v);
                float HG1 = tex2D<float>(_HGLut, u2, v);
                X_Val_Hg[i] = log(HG0 * HG1 + 1.0f);
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp =
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.z;
    AvgPool[POOL + POOL] = g
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.z;
    AvgPool[POOL + POOL] = g
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.z;
    AvgPool[POOL + POOL] = g;
    AvgPool[POOL + POOL + 1] = gamma
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.z;
    AvgPool[POOL + POOL] = g;
    AvgPool[POOL + POOL + 1] = gamma
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    sc
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0]
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.z;
    AvgPool[POOL + POOL] = g;
    AvgPool[POOL + POOL + 1] = gamma
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.z;
    AvgPool[POOL +
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.z;
    AvgPool[POOL + POOL] = g;
    AvgPool[POOL + POOL + 1] = gamma
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.z;
    AvgPool[POOL + POOL] = g;
    AvgPool
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.z;
    AvgPool[POOL + POOL] = g;
    AvgPool[POOL + POOL + 
```


## License: GPL-3.0
https://github.com/What-a-stupid-username/MRPNN/blob/3e949f801e3d920b99b58f041572d60779a64f7d/core/radiancePredict.cu

```
X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.z;
    AvgPool[POOL + POOL] = g;
    AvgPool[POOL + POOL + 1] = gamma
```

