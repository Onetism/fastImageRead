add_cython_target(_imreadfast CXX)
add_library(_imreadfast MODULE ${_imreadfast})
target_link_libraries(_imreadfast imagesread)
python_extension_module(_imreadfast)

install(TARGETS _imreadfast LIBRARY DESTINATION imreadfast)
install(FILES __init__.py DESTINATION imreadfast)