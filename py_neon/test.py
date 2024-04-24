import py_neon as ne

grid = ne.DGrid()
span_device_id0_standard = grid.get_span(ne.Execution.device,
                                         0,
                                         ne.DataView.standard)
print(span_device_id0_standard)

