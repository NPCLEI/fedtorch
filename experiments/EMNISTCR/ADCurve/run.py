import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO import *

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A1d00C62/run.py > lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/ADCurve/run.py
    # pkill -f gunicorn

    # exp.run([(ADCure,'.ADCure -workconfig --x_start "./PlotEMNISTCR/xstar_vs_xistar/A10C62.FedADAM"')],alpha=10,clientn=62)
    exp.run([(ADCure,'.ADCure -workconfig --x_start "./PlotEMNISTCR/xstar_vs_xistar/A0.001C62.FedADAM"')],alpha=0.001,clientn=62)
    exp.run([(ADCure,'.ADCure -workconfig --x_start "./PlotEMNISTCR/xstar_vs_xistar/A0.01C62.FedADAM"')],alpha=0.01,clientn=62)
    exp.run([(ADCure,'.ADCure -workconfig --x_start "./PlotEMNISTCR/xstar_vs_xistar/A0.1C62.FedADAM"')],alpha=0.1,clientn=62)
    exp.run([(ADCure,'.ADCure -workconfig --x_start "./PlotEMNISTCR/xstar_vs_xistar/A1C62.FedADAM"')],alpha=1,clientn=62)