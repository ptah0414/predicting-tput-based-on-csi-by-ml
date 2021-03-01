from bbtest_base import *
import socket
import random
import time
import pandas as pd

HOST = '127.0.0.1'
PORT = 9998
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))


@pytest.mark.timeout(0)
@pytest.mark.skipif(pytest.config.getoption('--bw') != 100 , reason="feature not supported for other bandwidths yet")
def test_ue_off_on_throughput_pdsch_csi_given_mcs(dlrx_1, ultx_1, bs_l2, ue_l2, cmdopt, dl_chemu, ssb=SSB6, bs_role='BS', ue_role='UE'):

	if ue_role != 'UE':
		ue = ue_l2
	else:
		ue = ue_l2(port=8020).get()
			
	dlrx_1.p.pdsch_rx.Plotting_slot().set(4)
		
	while True:
				
		dl_chemu.p.tp0.Path_loss_tp0_ue0().set(round(random.uniform(28.5, 30.5), 2))
		dl_chemu.p.xchan.Noise_power_ue0_rx().set(round(random.uniform(-51.5, -49.5), 2))
		
		msg1 = str(read_meas(dlrx_1.m.pdsch_rx.SINR_all_rx()))
		data = msg1.encode()
		client_socket.send(data)
		time.sleep(0.001)
		
		msg2 = str(read_meas(dlrx_1.m.srvcell_sss.sss_snr()))
		data = msg2.encode()
		client_socket.send(data)
		time.sleep(0.001)
		
		msg3 = str(read_meas(dlrx_1.m.srvcell_sss.sss_rsrp()))
		data = msg3.encode()
		client_socket.send(data)
		time.sleep(0.001)
		
		msg4 = str(read_meas(dlrx_1.m.srvcell_sss.sss_sinr()))
		data = msg4.encode()
		client_socket.send(data)
		time.sleep(0.001)
		
		msg5 = str(read_meas(ultx_1.m.csi_rx.Estimated_SINR()))
		data = msg5.encode()
		client_socket.send(data)
		time.sleep(0.001)
			
		msg6 = str(read_meas(ultx_1.m.csi_rx.Channel_Quality_Indicator()))
		data = msg6.encode()
		client_socket.send(data)
		
		
		data = client_socket.recv(64)
		msg = data.decode()
		num = float(msg)
		
		if num > 441:
			num == 441
		
		ue.p.uec.AI_estimated_tput().set(num)
		time.sleep(2)
	  
	client_socket.close();
	
