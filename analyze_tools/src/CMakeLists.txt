set(source_files main.cpp cell_linked_list.h cell_linked_list.cpp)

pybind11_add_module(analyze_tools ${source_files})