adaptive remeshing systems `dynamic_varlen` fields, since
their meshes actually get remeshed frame to frame and the current parser only supports
`static`/`dynamic`,.

also found a that `flag_dynamic_sizing`'s `sizing_field` has malformed `dtype` value
`"<dtype: 'float32'>"` instead of `"float32"`. Need to fix.

make ADR.md. keep engineering choices in there, including the ones ive made so far about
pre processing optimizations.