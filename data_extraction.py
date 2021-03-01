from bbtest_base import *
import random
import numpy as np
import csv
import os


@pytest.mark.timeout(0)
@pytest.mark.skipif(pytest.config.getoption('--bw') != 100 , reason="feature not supported for other bandwidths yet")
def test_ue_off_on_throughput_pdsch_csi_given_mcs(bs_l2, ue_l2, cmdopt, dl_chemu, ssb=SSB6, bs_role='BS', ue_role='UE'):
		
	if ue_role != 'UE':
		ue = ue_l2
	else:
		ue = ue_l2(port=8020).get()


	SINR_all_rx = []
	sss_snr = []
	sss_rsrp = []
	sss_sinr = []
	Estimated_SINR = []
	Channel_Quality_Indicator = []
	L1_rx_throughput_mbps = []
	
	f = open('meas.csv', 'a')	
	
	wr = csv.writer(f)

	num = 0
	for i in range(3000):
		num += 1

		print("")
		print('epoch: '+str(num))

		#rand_pathloss = round(random.uniform(29, 40), 2)
		#rand_noise = round(random.uniform(-56, -46), 2)
		rand_pathloss = round(random.uniform(31, 34.2), 2)
		rand_noise = round(random.uniform(-56, -53), 2)

		dl_chemu.p.tp0.Path_loss_tp0_ue0().set(rand_pathloss)
		dl_chemu.p.xchan.Noise_power_ue0_rx().set(rand_noise)
		
		SINR_all_rx = read_meas(dlrx_1.m.pdsch_rx.SINR_all_rx())
		sss_snr = read_meas(dlrx_1.m.srvcell_sss.sss_snr())
		sss_rsrp = read_meas(dlrx_1.m.srvcell_sss.sss_rsrp())
		sss_sinr = read_meas(dlrx_1.m.srvcell_sss.sss_sinr())
		Estimated_SINR = read_meas(ultx_1.m.csi_rx.Estimated_SINR())
		Channel_Quality_Indicator = read_meas(ultx_1.m.csi_rx.Channel_Quality_Indicator())

		time.sleep(2)
		L1_rx_throughput_mbps = read_meas(ue.m.L1_rx_throughput_mbps())
			
	
		SINR_all_rx.append(SINR_all_rx)
		sss_snr.append(sss_snr)
		sss_rsrp.append(sss_rsrp)
		sss_sinr.append(sss_sinr)
		Estimated_SINR.append(Estimated_SINR)
		Channel_Quality_Indicator.append(Channel_Quality_Indicator)
		L1_rx_throughput_mbps.append(L1_rx_throughput_mbps)

	
		f.write(str(SINR_all_rx[i]) + ',' + str(sss_snr[i]) + ',' 
		+ str(sss_rsrp[i]) +  str(sss_sinr[i]) + ',' 
		+ str(Estimated_SINR[i]) + ',' + str(Channel_Quality_Indicator[i]) + ','
		+ str(L1_rx_throughput_mbps[i]) +'\n')
		
	f.close()	
