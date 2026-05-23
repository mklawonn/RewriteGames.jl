"""
Multi-word (chunked) bitset helpers for CSP domains with >64 elements.

Convention: chunk c (1-based) covers elements (c-1)*64+1 through c*64.
Bit b (0-based) within chunk c represents element (c-1)*64 + b + 1.

Flat domain layout for n_vars variables with n_chunks chunks each:
  domains[(v-1)*n_chunks + c]  =  chunk c of variable v's domain.

Flat hom-forward layout for n_elems elements with n_chunks chunks:
  hom_fwd[(w-1)*n_chunks + c]  =  chunk c of the bitmask for element w.

MAX_CHUNKS caps the number of chunks for static allocation in GPU kernels.
Increase it to support more than MAX_CHUNKS * 64 elements per type.
"""

const MAX_CHUNKS = 4   # default; overridden per-rule via _select_nc_max

function _select_nc_max(nc::Int)
    nc <= 1  && return 1
    nc <= 2  && return 2
    nc <= 4  && return 4
    nc <= 8  && return 8
    nc <= 16 && return 16
    error("nc=$(nc) exceeds maximum supported chunks (16 = 1024 elements/type)")
end

# Convert 1-based element index to (chunk_idx, bit_idx)
@inline function elem_to_chunk(i::Int)
    ci = ((i - 1) >> 6) + 1
    bi = (i - 1) & 63
    (ci, bi)
end

# Convert (chunk_idx, bit_idx) back to 1-based element index
@inline function chunk_to_elem(ci::Int, bi::Int)
    (ci - 1) * 64 + bi + 1
end

@inline function mw_iszero(d::AbstractVector{UInt64}, nc::Int)
    for c in 1:nc
        d[c] != UInt64(0) && return false
    end
    return true
end

@inline function mw_count_ones(d::AbstractVector{UInt64}, nc::Int)
    s = 0
    for c in 1:nc
        s += count_ones(d[c])
    end
    s
end

# Returns (chunk_idx, bit_idx) of lowest set bit (both 1-based chunk, 0-based bit),
# or (0, 0) if all zero
@inline function mw_first_bit(d::AbstractVector{UInt64}, nc::Int)
    for c in 1:nc
        d[c] != UInt64(0) && return (c, Int(trailing_zeros(d[c])))
    end
    return (0, 0)
end

# Build a full-domain mask for n_elems elements with n_chunks chunks
function mw_full_mask(n_elems::Int, nc::Int)
    mask = zeros(UInt64, nc)
    for i in 1:n_elems
        ci, bi = elem_to_chunk(i)
        ci <= nc && (mask[ci] |= UInt64(1) << bi)
    end
    mask
end
