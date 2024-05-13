import py_ne as ne

grid = ne.Dense_grid()
span_device_id0_standard = grid.get_span(ne.Execution.device,
                                         0,
                                         ne.Data_view.standard)
print(span_device_id0_standard)

