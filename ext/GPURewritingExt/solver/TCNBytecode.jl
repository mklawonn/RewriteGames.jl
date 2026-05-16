"""
    TCNBytecode

A 16-byte constraint packet for the Turbo GPU solver.

Layout (all little-endian):
  bytes  0-1   op     UInt16   operation code
  bytes  2-3   var1   UInt16   primary variable index (1-based, 0 = unused)
  bytes  4-5   var2   UInt16   secondary variable index
  bytes  6-7   var3   UInt16   auxiliary/reification variable
  bytes  8-11  param1 Int32    integer parameter (domain value, hom index, …)
  bytes 12-15  param2 Int32    integer parameter

Variable indices address the flat variable array: one slot per element
position in the pattern L, laid out as
  [ob₁-elems …, ob₂-elems …, …, ob_k-elems …]
in the order given by `SchemaInfo.obj_types`.
"""
struct TCNBytecode
    op     :: UInt16
    var1   :: UInt16
    var2   :: UInt16
    var3   :: UInt16
    param1 :: Int32
    param2 :: Int32
end

# Verify the struct is exactly 16 bytes (critical for GPU memory layout)
@assert sizeof(TCNBytecode) == 16 "TCNBytecode must be exactly 16 bytes"

# ── Operation codes ────────────────────────────────────────────────────────────

"""
`PROP_FUNC`: morphism propagation constraint.
  var1 must map to the image of var2 under the morphism encoded in param1.
  Prunes the domain of var1 to values reachable from current domain of var2.
"""
const PROP_FUNC  = UInt16(0x0001)

"""
`PROP_EQ`: equality constraint between two pattern variables.
  var1 and var2 must be assigned the same world element.
"""
const PROP_EQ    = UInt16(0x0002)

"""
`PROP_NEQ`: inequality constraint.
  var1 and var2 must be assigned distinct world elements (monic constraint).
"""
const PROP_NEQ   = UInt16(0x0003)

"""
`PROP_ATTR_EQ`: attribute equality constraint.
  var1's assignment must have attribute column `param1` equal to the
  encoded integer value `param2`.
"""
const PROP_ATTR_EQ = UInt16(0x0004)

"""
`PROP_ATTR_LEQ`: attribute inequality constraint (ordinal only).
  var1's attribute column `param1` must be ≤ `param2`.
"""
const PROP_ATTR_LEQ = UInt16(0x0005)

"""
`NAC_REIF`: reified NAC sub-constraint.
  var3 is a boolean auxiliary variable. If var3 == 1 the NAC is matched
  (forbidden); propagation prunes world states where var3 could be forced to 1.
  param1 = NAC group ID (all bytecodes with the same group ID form one NAC).
"""
const NAC_REIF   = UInt16(0x0010)

"""
`PAC_REIF`: reified PAC sub-constraint (required application condition).
  Dual of NAC_REIF: param1 = PAC group ID.
"""
const PAC_REIF   = UInt16(0x0011)

"""
`DOMAIN_SIZE`: meta-constraint bounding the domain of var1 to [1..param1].
  Emitted once per variable at the start of the bytecode array.
"""
const DOMAIN_SIZE = UInt16(0x0020)

"""
    tcn(op, var1=0, var2=0, var3=0, param1=0, param2=0) -> TCNBytecode

Convenience constructor with keyword-defaulted fields.
"""
tcn(op; var1::Integer=0, var2::Integer=0, var3::Integer=0,
        param1::Integer=0, param2::Integer=0) =
    TCNBytecode(UInt16(op), UInt16(var1), UInt16(var2), UInt16(var3),
                Int32(param1), Int32(param2))
