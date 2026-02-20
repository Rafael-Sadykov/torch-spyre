# Kernel Tile IR

**Authors:**
* @MoriOhara
* @TakuyaNakaike
* @Prasanth
* @swagath.venkataramani
* @ishizaki
* @erio

## **Summary**
This RFC proposes an internediate representation as an MLIR dialect to represent kernels as an interface between a frontend and a backend of PyTorch compiler for Spyre.

## **Motivation**
- To enable custom or hand-written kernels optimized for performance on Spyre, which are critical for inferencing performance.
- To accelerate the compiler development by leveraging open-source technologies, such as the MLIR framework.
- To define a kernel representation tailered for dataflow-oriented AI acceleratrors, such as Spyre.

## **Proposed Implementation**

This dialect represents a kernel tile, which defines to a series of operations against one or more data tiles to run on a single Sprye core. The data tile represents a unit of data with a shape to load, store, or extract a sub tile from. It is associated with a memory space, such as a device memory or a core-local memory (aka LX), or a composite tile spread across multiple memory spaces (aka distributed tiles). 

This dialect provides a method to define a tile with its layout information, to define a composite tile, and to extract a sub tile from a tile or a composite tile at given indexes. A sub tile access from a base tile is direct when the address of the sub tile is directly obtained from the address of the base tile and the indexes given or indirect when it is obtained through one or more other tiles to map the given indexes to the actual address offset to access each element in the sub tile.

This MLIR layer is intended to play as an interface between a frontend and a backend of PyTorch compiler where the frontend (Torch Inductor) partitions the workload across cores and tiles, while the backend optimizes/generates binaries to be executed at each execution unit inside a core. This layer exposes tile locations and movements across core-local and device memory spaces while it encapsurates details in arithmetic operations, such as vector and systolic array operations. This dialect relies on other standard MLIR dialects, as listed blow, to take advantage of their existing libraries.

- Dependent MLIR dialects
    - affine
    - arith
    - math
    - memref
    - ptr
    - scf
    - tensor
    - linalg

- Operations
    - [kt.arange](#ktarange)
    - [kt.composite_tile_view](#ktcomposite_tile_view)
    - [kt.func](#ktfunc)
    - [kt.grid_index](#ktgrid_index)
    - [kt.layout](#ktlayout)
    - [kt.load](#ktload)
    - [kt.offset_range](#ktoffset_range)
    - [kt.store](#ktstore)
    - [kt.tile_access](#kttile_access)
    - [kt.tile_grid_access](#kttile_grid_access)
    - [kt.tile_indirect_access](#kttile_indirect_access)
    - [kt.tile_view](#kttile_view)

- Examples
    - [add](#add)
    - [index_copy](#index_copy)
    - [sum](#sum)
    - [paged_attention](#paged_attention)

### Operations

#### kt.arange

The `kt.arange` operation returns a 1D tensor of `i32` values starting from `start` and ending at `end`.

Syntax:
```
operation ::= kt.arange start, end : tensor
```

Example:
```
%offset0 = kt.arange %c0, %c256 -> tensor<256xi32>
```

#### kt.compoiste_tile_view

The `kt.composite_tile_view` operation defines a distributed tile, which is a tile composed from multiple sub-tiles located at a different memory space, where each sub-tile is provided as a tile view defined by a [kt.tile_view](#kttile_view) operation.

Syntax:
```
operation ::= kt.composite_tile_view tileViews : [tileref] -> tileref
tileViews ::= [tileView0{,tileView1{,...}}]
```

Example:
```
%composite_tile_view = kt.composite_tile_view [%tile_view0, %tile_view1] : tileref<16x64xf16>, tileref<16x64xf16> -> tileref<2x16x64xf16>
```

#### kt.func

The `kt.func` operation defines a kernel function executed on a Spyre core. It can take one or more pointers on a memory space and possibly other arguments. It also takes the size of a grid space as an attribute, where each Spyre core is assigned to one grid point in the space.

limitation: the current implementation uses func.func for kt.func.

Syntax:
```
operation ::= kt.func@funcName(argument-list) -> returnType attributes { grid = gridSizes } {
    KernelBody
    return returnValue : returnType
}
argument-list ::= | argument{,argument-list}
argument ::= argumentName : argumentType 
gridSize ::= [sizeAtDim0{,sizeAtDim1{,sizeAtDim2}}]
```

Example:
```
kt.func @elementwise_add(%base_in0_hbm : index, %base_in1_hbm : index, %base_out_hbm : index) -> index attributes { grid = [16,2] } {
    ...
    return $c0 : index
}
```

#### kt.grid_index

The `kt.grid_index` operation provides the grid index as a scalar constant at a given grid dimension to run the kernel. The dimension specifies one of the dimentions in the grid space, which can be up to three. It returns the index as an integer value.

Syntax:
```
operation ::= kt.grid_index { dim = dimensionValue : i32 } : i32
dimensionValue ::= 0 | 1 | 2
```

Example:

```
%grid_index_dim0 = kt.grid_index { dim = 0 : i32 } : i32
```

#### kt.layout

The `kt.layout` attribute defines a memory layout of a tile in a context of the stick dimension, which consists of a list tile dimensions in a stick and the number of elements for each of the dimensions in the stick.
The `kt.tile_view` operation takes this layout to defines a physical memory layout of the tile. The stick dimensions are an inner dimension of a tile on a physical memory, where the hardware can access their consecutive elements efficiently. The stick corresponds to the minimum memory block that the hardware can access, which is 128 bytes for the current Spyre hardware. It can hold 64 of fp16 elements.

Syntax:
```
attribute ::= #st.layout<{stick_size=[numElemsAtStickDim0{,numElemsAtStickDim1}], stick_dim=[stickDim0{,stickDim1}]}>
stickDim0 ::= the innermost dimension in a stick
stickDim1 ::= the second innermost dimension in a stick
```

Example:
```
#1 = #kt.layout<{stick_size=[64], sitck_dim=[1]}>  // [0, 1]: 1 is innermost
```

#### kt.load

The `kt.load` operation loads a tile at a memory location specified by `srcTileRef` to `dstTileRef` when the destination tileref is given, or to another memory space which is determined by a backend compiler. This operation returns the destination tileref where the tensor object is loaded for the first case, or a tensor that corresponds to the loaded tile for the second case.

Syntax:
```
operation ::= kt.load srcTileRef, dstTileRef : tileref, tileref -> tileref
 | kt.load srcTileRef : tileref -> tensor
```

Example:
```
%tile_ref = kt.load %src_tile_ref, %dst_tile_ref : tileref<32x64xf16>, tileref<32x64xf16> -> tileref<32x64xf16>
%tile = kt.load %tile_ref : tileref<32x64xf16> -> tensor<32x64xf16>
```

#### kt.offset_range

The `kt.offset_rage` operation computes a tensor to represent a set of offsets to be used for indrect tile accesses. It takes an affine map with one or more 1-dimensional tensors and/or scalar values. The map is applied to those inputs to produce a tensor of offsets. An input tensor can be an index tensor obtained from an index tile on memory where the map maps each tensor index to an address offset. Alternatively, an input tensor can be a range of tensor indexes computed from [kt.arange](#kt.arange) where the map maps each tensor index with one or more optional scalar inputs to an address offset. These output tensors are intended to be used for indirect tile accesses though [kt.tile_indirect_access](#kt.tile_indirect_access).

Syntax:
```
operation ::= kt.offset_range #map (tensor) : tensorType,... -> tensorType
  | kt.offset_range #map (scalar|tensor,...) : i32|tensorType,... -> tensorType
```

Example:
```
#map1 = affine_map<(d0) -> (d0 * 1024)>
%offset_0 = kt.offset_range #map1 (%index_coretile) : tensor<128xi32> -> tensor<128xi32>

%arange_0_256 = kt.arange %c0, %c256 : tensor<256xi32>
#map2 = affine_map<(d0, d1) : (d0 * 256 + d1)>
%offset_1 = kt.offset_range #map2 (%index_1, %arange_0_256) : i32, tensor<256xi32> -> tensor<256xi32>
```

#### kt.store

The `kt.store` operation stores a tensor as a `srcTile` object or a tile at `srcTileRef` to a tile at `dstTileRef`. It retuns the destination tileref where the tensor object is stored.

Syntax:
```
operation ::= kt.store srcTile, dstTileRef : tensor, tileref -> tileref
 | kt.store srcTileRef, dstTileRef : tileref, tileref -> tileref
```

Example:
```
%tileRef = kt.store %tile, %dstTileRef : tensor<32x64xf16>, tileref<32x64xf16> -> tileref<32x64xf16>
%tileRef = kt.store %srcTileRef, %dstTileRef : tileref<32x64xf16>, tileref<32x64xf16> -> tileref<32x64xf16>
```

#### kt.tile_access

The `kt.tile_access` operation extracts a sub tile from `srcTileRef` at `indexes` to return a `tileref` object as a reference to the sub tile. The sub tile shape is given by the `access_tile_set` attribute.

Syntax:
```
operation ::= kt.tile_access srcTileRef, indexes {
    access_tile_set = accessTileSetAttr : IntegerSetAttr
} : tileref, [i32] -> tileref
indexes := [index0{,index1{,index2}}]
```

Example:
```
%a_coretile_ref = kt.tile_access %a_tile_ref, [%x_index, %y_index] { 
    access_tile_set = #affine_set<(d0, d1) : (0 <= d0 < 128, 0 <= d1 < 256)>
} : tileref<512x1024xf16>, [i32, i32] -> tileref<128x256xf16>
```

#### kt.tile_grid_access

The `kt.tile_grid_access` operation extracts a sub tile from `srcTileRef` at `gridIndexes` to return a `tileref` object as a reference to the sub tile. The sub tile shape is given by the `access_tile_set` attribute. This is similar to [kt.tile_access](#kttile_access), which takes indexes to extract a tile, while this operation takes grid indexes to slice a tile for each dimension with its grid size.

Syntax:
```
operation ::= kt.tile_access srcTileRef, gridIndexes {
    access_tile_set = accessTileSetAttr : IntegerSetAttr
} : tileref, [i32] -> tileref
indexes := [index0{,index1{,index2}}]
```

Example:
```
%a_coretile_ref = kt.tile_grid_access %a_tile_ref, [%grid0, %grid1] { 
    access_tile_set = #affine_set<(d0, d1) : (0 <= d0 < 128, 0 <= d1 < 256)>
} : tileref<512x1024xf16>, [i32, i32] -> tileref<128x256xf16>
```

#### kt.tile_indirect_access

The `kt.tile_indirect_access` operation extracts a sub tile from `srcTileRef` where the address offset of each element is obtained from `offsetTensors` to return a `tileref` object as a reference to the sub tile. Each offset tensor `offsetTensor` is a block offset for a dimension of `srcTileRef`. The offset tensor is typically obtained from [kt.offset_ranges](#kt.offset_ranges) with an index tensor extracted from an index tile or an index range obtained from [kt.arange](kt.arange).

Each element in the offset tensor is typically obtained from an index of a dimension multiplied by the stride of the dimension. The index value is typically obtained from an index tile for an indirectly indexed dimension or from a range of indexes for a directly indexed dimension.

Syntax:
```
operation ::= kt.tile_indirect_access srcTileRef, [offsetTensors] : tilerefType, [tensorTypes] -> tilerefType
```

Example;
```
%grid0 = kt.grid_index {dim = 0 : i32} -> i32
%grid1 = kt.grid_index {dim = 1 : i32} -> i32

#set4 = affine_set<(d0) : (0 <= d0 < 128)>
%index_coretile_ref = kt.tile_grid_access %index_tile_ref, [%grid0] { access_tile_set = #set4 } : memref<512xi32>, [i32] -> tileref<128xi32>
%index_coretile = kt.load %index_coretile_ref : tileref<128xi32> -> tensor<128xi32>

#map1 = affine_map<(d0) -> (d0 * 1024)>
%offset_0 = kt.offset_range #map1 (%index_coretile) : tensor<128xi32> -> tensor<128xi32>
%arange_0_256 = kt.arange %c0, %c256 : tensor<256xi32>
#map2 = affine_map<(d0, d1) : (d0 * 256 + d1)>
%offset_1 = kt.offset_range #map2 (%grid1, %arange_0_256) : i32, tensor<256xi32> -> tensor<256xi32>

%coretile_ref = kt.tile_indirect_access %base_tile_ref, [%offset_0, %offset_1] : tileref<512x1024xf16>, [tensor<128xi32>, tensor<256xi32>] -> tileref<128x256xf16>

```

#### kt.tile_view

The `kt.tile_view` operation defines a physical memory layout of a tile at a location given by `ptr` on a memory space to return a `tileref` object, associated with a shape and strides given by the `coordinate_set` and `strides` attributes, respectively. The memory space (LX or device memory), where the tile is located, is given by the `memory_space` attribute. The stick layout, inner-most dimensions on the physical memory layout, is given by an [kt.layout](#ktlayout) attribute.

limitation: the current implementation uses an index for ptr.

Syntax:
```
operation ::= kt.tile_view ptr {
    coordinate_set = coordinateSetAttr : IntegerSetAttr,
    strides = strideValues : [index],
    memory_space = memorySpaceID : index,
    layout = layoutAttr : kt.LayoutAttr
} : index -> tileref
strideValues := [stride0{,stride1{,stride2}}]
```

Example:
```
%a_tile_ref = kt.tile_view %a_ptr {
    coordinate_set = #affine_set<(d0, d1) : (0 <= d0 < 512, 0 <= d1 < 1024)>,
    strides = [1024, 1],
    memory_space = HBM,
    layout = #kt.layout<stick_size=[64], stick_dim=[1]>
} : index -> tileref<512x1024xf16>
```


### Examples

#### add

```mlir
// OriginalTile: [512, 1024]
// CoreTile: [128, 256] * 4 * 4
#set1 = affine_set<(d0, d1) : (0 <= d0 < 512, 0 <= d1 < 1024)>
#set2 = affine_set<(d0, d1) : (0 <= d0 < 128, 0 <= d1 < 256)>
#set3 = affine_set<(d0, d1) : (0 <= d0 < 32, 0 <= d1 < 64)>
module {
  func.func @add(%a_ptr: index, %b_ptr: index, %c_ptr: index) -> index attributes { grid = [4, 4] } {
    %c0_index = aith.constant 0 : index

    // Original tiles
    // #set1 = affine_set<(d0, d1) : (0 <= d0 < 512, 0 <= d1 < 1024)>
    %a_tile_ref = kt.tile_view %a_ptr {
      coordinate_set = #set1,
      strides = [1024, 1],
      memory_space = HBM,
      layout = #kt.layout<stick_size=[64], stick_dim=[1]>
    } : index -> tileref<512x1024xf16>
    %b_tile_ref = kt.tile_view %b_ptr {
      coordinate_set = #set1,
      strides = [1024, 1],
      memory_space = HBM,
      layout = #kt.layout<stick_size=[64], stick_dim=[1]>
    } : index -> tileref<512x1024xf16>
    %c_tile_ref = kt.tile_view %c_ptr {
      coordinate_set = #set1,
      strides = [1024, 1],
      memory_space = HBM,
      layout = #kt.layout<stick_size=[64], stick_dim=[1]>
    } : index -> tileref<512x1024xf16>

    %grid0 = kt.grid_index { dim = 0 : i32 } -> i32
    %grid1 = kt.grid_indexå { dim = 1 : i32 } -> i32

    // #set2 = affine_set<(d0, d1) : (0 <= d0 < 128, 0 <= d1 < 256)>
    %a_coretile_ref = kt.tile_grid_access %a_tile_ref, [%grid0, %grid1] { access_tile_set = #set2 } : tileref<512x1024xf16>, [i32, i32] -> tileref<128x256xf16>
    %b_coretile_ref = kt.tile_grid_access %b_tile_ref, [%grid0, %grid1] { access_tile_set = #set2 } : tileref<512x1024xf16>, [i32, i32] -> tileref<128x256xf16>
    %c_coretile_ref = kt.tile_grid_access %c_tile_ref, [%grid0, %grid1] { access_tile_set = #set2 } : tileref<512x1024xf16>, [i32, i32] -> tileref<128x256xf16>

    %a_coretile = kt.load %a_coretile_ref : tileref<128x256xf16> -> tensor<128x256xf16>
    %b_coretile = kt.load %b_coretile_ref : tileref<128x256xf16> -> tensor<128x256xf16>

    %c_coretile = arith.addf %a_coretile, %b_coretile : tensor<128x256xf16>, tensor<128x256xf16> -> tensor<128x256xf16>

    kt.store %c_coretile, %c_coretile_ref : tensor<128x256xf16>, tileref<128x256xf16>

    return %c0_index : index
  }
}
```

#### index_copy

```mlir
// OriginalTile: [512, 1024], OriginalIndexTile: [512]
// CoreTile: [128, 256] * 4 * 4, IndexCoreTile: [128] * 4
#set1 = affine_set<(d0, d1) : (0 <= d0 < 512, 0 <= d1 < 1024)>
#set2 = affine_set<(d0) : (0 <= d0 < 512)>
#set3 = affine_set<(d0, d1) : (0 <= d0 < 128, 0 <= d1 < 256)>
#set4 = affine_set<(d0) : (0 <= d0 < 128)>
#map1 = affine_map<(d0) -> (d0 * 1024)>
#map2 = affine_map<(d0, d1) : (d0 * 256 + d1)>
module {
  func.func @indexcopy(%src_ptr: index, %dst_ptr: index, %index_ptr: index) -> index attributes { grid = [4, 4] }  {
    %c0_index = arith.constant 0 : index
    %c256 = arith.constant 256 : i32

    // Original tiles
    // #set1 = affine_set<(d0, d1) : (0 <= d0 < 512, 0 <= d1 < 1024)>
    %src_tile_ref = kt.tile_view %src_ptr {
      coordinate_set = #set1,
      strides = [1024, 1],
      memory_space = HBM,
      layout = #kt.layout<stick_size=[64], stick_dim=[1]>
    } : index -> tileref<512x1024xf16>
    %dst_tile_ref = kt.tile_view %dst_ptr {
      coordinate_set = #set1,
      strides = [1024, 1],
      memory_space = HBM,
      layout = #kt.layout<stick_size=[64], stick_dim=[1]>
    } : index -> tileref<512x1024xf16>
    // #set2 = affine_set<(d0) : (0 <= d0 < 512)>
    %index_tile_ref = kt.tile_view %index_ptr {
      coordinate_set = #set2,
      strides = [1],
      memory_space = HBM,
      layout = #kt.layout<stick_size=[32], stick_dim=[1]>
    } : index -> tileref<512xi32>

    %grid0 = kt.grid_index {dim = 0 : i32} -> i32
    %grid1 = kt.grid_index {dim = 1 : i32} -> i32

    // #set3 = affine_set<(d0, d1) : (0 <= d0 < 128, 0 <= d1 < 256)>
    %src_coretile_ref = kt.tile_grid_access %src_tile_ref, [%grid0, %grid1] { access_tile_set = #set3 } : tileref<512x1024xf16>, [i32, i32] -> tileref<128x256xf16>
    %src_coretile = kt.load %src_coretile_ref : tileref<128x256xf16> -> tensor<128x256xf16>

    // #set4 = affine_set<(d0) : (0 <= d0 < 128)>
    %index_coretile_ref = kt.tile_grid_access %index_tile_ref, [%grid0] { access_tile_set = #set4 } : memref<512xi32>, [i32] -> tileref<128xi32>
    %index_coretile = kt.load %index_coretile_ref : tileref<128xi32> -> tensor<128xi32>

    // #map1 = affine_map<(d0) -> (d0 * 1024)>
    %offset_0 = kt.offset_range #map1 (%index_coretile) : tensor<128xi32> -> tensor<128xi32>
    %arange_0_256 = kt.arange %c0, %c256 : tensor<256xi32>
    // #map2 = affine_map<(d0, d1) : (d0 * 256 + d1)>
    %offset_1 = kt.offset_range #map2 (%grid1, %arange_0_256) : i32, tensor<256xi32> -> tensor<256xi32>

    %dst_coretile_ref = kt.tile_indirect_access %dst_tile_ref, [%offset_1, %offset_0] : tileref<512x1024xf16>, [tensor<128xi32>, tensor<256xi32>] -> tileref<128x256xf16>

    kt.store %src_coretile, %dst_coretile_ref : tensor<128x256xf16>, tileref<128x256xf16>

    return %c0_index : index
  }
}
```

#### sum

```mlir
Coming soon
```

#### paged_attention

```mlir
Coming soon
```

## **Metrics **
TBD

## **Drawbacks**
The frontend compiler (torch-spyre) needs to be enhanced to generate kernels in KTIR, and the backend compiler needs to be enhanced to compile them.

## **Alternatives**
What other designs have been considered? What is the impact of not doing this?

## **Prior Art**
The frontend compiler currently uses Super DSC as an intermediate representation (IR) of kernels to be compiled by the backend. The format is proprietary and very different from industry standard IRs, such as MLIR dialects. It is difficult to represent hand-written kernels, which are critical for inferencing performance.

<!--
## **How we teach this**
* What names and terminology work best for these concepts and why? How is this idea best presented?
* Would the acceptance of this proposal mean the PyTorch documentation must be re-organized or altered?
* How should this feature be taught to existing PyTorch users?

## **Unresolved questions**
* What parts of the design do you expect to resolve through the RFC process before this gets merged?
* What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
* What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?

## Resolution
We decided to do it. X% of the engineering team actively approved of this change.

### Level of Support
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.

#### Additional Context
Some people were in favor of it, but some people didn’t want it for project X.
-->

### Next Steps
Will implement it.

#### Tracking issue
https://github.com/torch-spyre/torch-spyre/issues/663

<!--
#### Exceptions
Not implementing on project X now. Will revisit the decision in 1 year.
-->
