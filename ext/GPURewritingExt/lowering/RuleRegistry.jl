using CUDA

"""
    DeviceRuleRegistry

Flat, GPU-resident registry of all rewrite rules in a schedule.
"""
struct DeviceRuleRegistry{BV, OV, NV, RV, HV, AV}
    csp_bytecodes    :: BV
    csp_offsets      :: OV
    csp_lens         :: OV
    csp_n_vars       :: NV
    
    rhs_n_add_flat   :: RV
    rhs_hom_data     :: HV
    rhs_hom_offsets  :: OV
    rhs_attr_data    :: AV
    rhs_attr_offsets :: OV
end

import Adapt: adapt_structure
function adapt_structure(to, reg::DeviceRuleRegistry)
    DeviceRuleRegistry(
        adapt_structure(to, reg.csp_bytecodes),
        adapt_structure(to, reg.csp_offsets),
        adapt_structure(to, reg.csp_lens),
        adapt_structure(to, reg.csp_n_vars),
        adapt_structure(to, reg.rhs_n_add_flat),
        adapt_structure(to, reg.rhs_hom_data),
        adapt_structure(to, reg.rhs_hom_offsets),
        adapt_structure(to, reg.rhs_attr_data),
        adapt_structure(to, reg.rhs_attr_offsets)
    )
end
