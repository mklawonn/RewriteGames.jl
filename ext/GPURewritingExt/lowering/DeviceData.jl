using CUDA

"""
    DeviceACSet

GPU-native flattened representation of an ACSet world for use in kernels.
"""
struct DeviceACSet{AV, NV, OV, HV, HOV, COV, ATV, AOV}
    active          :: AV # CuVector{Bool}
    n_live          :: NV # CuVector{Int32}
    obj_offsets     :: OV # CuVector{Int32}
    n_alloc         :: OV # CuVector{Int32}
    
    homs            :: HV # CuVector{Int32}
    hom_offsets     :: HOV # CuVector{Int32}
    hom_cod_offsets :: COV # CuVector{Int32}
    
    attrs           :: ATV # CuVector{Int32}
    attr_offsets    :: AOV # CuVector{Int32}
end

import Adapt: adapt_structure
function adapt_structure(to, g::DeviceACSet)
    DeviceACSet(
        adapt_structure(to, g.active),
        adapt_structure(to, g.n_live),
        adapt_structure(to, g.obj_offsets),
        adapt_structure(to, g.n_alloc),
        adapt_structure(to, g.homs),
        adapt_structure(to, g.hom_offsets),
        adapt_structure(to, g.hom_cod_offsets),
        adapt_structure(to, g.attrs),
        adapt_structure(to, g.attr_offsets)
    )
end
