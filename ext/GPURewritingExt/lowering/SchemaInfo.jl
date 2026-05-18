"""
    SchemaInfo

Flat, GPU-serialisable snapshot of a Catlab ACSet schema. Built once from a
world ACSet on the host, then threaded through every lowering and kernel call.
"""
struct SchemaInfo
    obj_types :: Vector{Symbol}         # ob(S)
    homs      :: Vector{Symbol}         # hom(S)
    attrs     :: Vector{Symbol}         # attr(S)
    hom_dom   :: Dict{Symbol, Symbol}   # morphism → domain obj type
    hom_cod   :: Dict{Symbol, Symbol}   # morphism → codomain obj type
    attr_dom  :: Dict{Symbol, Symbol}   # attribute → owner obj type
    obj_index :: Dict{Symbol, Int}      # obj type → dense integer index
end

"""
    extract_schema_info(world) -> SchemaInfo

Inspect the schema of `world` (any ACSet) and return a `SchemaInfo`.
"""
function extract_schema_info(world)
    S         = acset_schema(world)
    obj_types = collect(Symbol, ob(S))
    homs_list = collect(Symbol, hom(S))
    attrs_list = collect(Symbol, attr(S))

    hom_dom  = Dict{Symbol,Symbol}()
    hom_cod  = Dict{Symbol,Symbol}()
    for h in homs_list
        hom_dom[h] = dom(S, h)
        hom_cod[h] = codom(S, h)
    end

    attr_dom = Dict{Symbol,Symbol}()
    for a in attrs_list
        attr_dom[a] = dom(S, a)
    end

    obj_index = Dict{Symbol,Int}(o => i for (i,o) in enumerate(obj_types))

    SchemaInfo(obj_types, homs_list, attrs_list, hom_dom, hom_cod, attr_dom, obj_index)
end
