rd /s /q Z:\GO-test-run-leveldb
#rd /s /q F:\caffe-windows\examples\GO\GO-test-run-leveldb

del  F:\caffe-windows\examples\GO\test-run\convert_GO_data.exe.DESKTOP-3POG0OV.teluw.log* /q
; ��·������һ�� ��mutiGO ��sgf �ļ� ������test-run ��1 Ϊ�׷� 2 Ϊ�ڷ�
F:\caffe-windows\examples\GO\convert_GO_data.exe  F:\caffe-windows\examples\GO\test-run 1 19999 Z:\GO-test-run-leveldb --backend=leveldb


.\3rdparty\bin\caffe.exe test --debug=true --iterations=26 --weights=examples/GO/lenet_7x7_iter_22000.caffemodel --model=examples/GO/GO_demo_LeNet_test_pool_run.prototxt

pause
