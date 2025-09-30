function [ACP_L,ACP_H]=Rf_Spectrum_Cntrl_N9042B(Set_flag,Freq,ACP_TXCH_BAND,TX_pairs,TX_Space,ACP_ADJ_BAND_SPAC,rbw,ref_lev)
%__author__ = "Yizhuo Wu"
%__license__ = "Apache-2.0 License"
%__email__ = "yizhuo.wu@tudelft.nl"

%Rf_Spectrum_Cntrl_N9042B - Keysight N9042B Spectrum Analyzer Control Function
%This function provides the same functionality as Rf_Spectrum_Cntrl_FSW.m
%but is designed for Keysight N9042B signal analyzer
%
%Input parameters:
%   Set_flag: 1 for setup and measurement, 0 for measurement only
%   Freq: Center frequency in Hz
%   ACP_TXCH_BAND: TX channel bandwidth in MHz
%   TX_pairs: Number of TX channel pairs
%   TX_Space: TX channel spacing in MHz
%   ACP_Adj_pairs: Number of adjacent channel pairs
%   ACP_ADJ_BAND: Adjacent channel bandwidth in MHz
%   ACP_ADJ_BAND_SPAC: Adjacent channel spacing in MHz
%   Ns, iter, fc, rel, BW, fs: Additional parameters (for compatibility)
%
%Output parameters:
%   P_DBM: Channel power in dBm
%   ACP_L: Lower adjacent channel power in dBc
%   ACP_H: Upper adjacent channel power in dBc

% Create VISA connection to Keysight N9042B
% Note: Replace the IP address with your instrument's IP address
V = visa('agilent','GPIB0::18::INSTR');
fopen(V)
% Alternative connection methods (uncomment as needed):
% V = visadev('TCPIP0::192.168.1.100::5025::SOCKET');  % Socket connection
% V = visadev('USB0::0x2A8D::0x1B0B::MY12345678::0::INSTR');  % USB connection

% Instrument Configuration and Control
if(Set_flag==0)
    % Query instrument identification
    idn = query(V, '*IDN?');
    disp(idn);
    
    FQ_A = Freq;
    
    % Configure input attenuation (0 dB)
    fprintf(V, 'SENS:POW:RF:ATT 10');
    fprintf(V, 'INST:CONF:SA:ACP');
    % Set center frequency
    FREQ_Str = sprintf('SENS:FREQ:CENT %.1f', FQ_A);
    fprintf(V, FREQ_Str);
    
    % Configure ACP measurement
    % Set number of TX channel pairs
    TX_Pairs = sprintf('SENS:ACP:CARRier1:COUN %d', TX_pairs);
    fprintf(V, TX_Pairs);
     % Set TX channel bandwidth
    ACP_TXCH_BAND_C = sprintf('SENS:ACP:CARRier1:LIST:BAND:INT %dMHz', ACP_TXCH_BAND);
    fprintf(V, ACP_TXCH_BAND_C);

    % Set TX channel spacing
    TX_Space_Str = sprintf('SENS:ACP:CARRier1:LIST:WIDT %dMHz', TX_Space);
    fprintf(V, TX_Space_Str);
    
    
    % Set adjacent channel bandwidth
    ACP_ADJ_BAND_C = sprintf('SENS:ACP:OFFSet1:OUT:LIST:STAT 1,0');
    fprintf(V, ACP_ADJ_BAND_C);

    ACP_ADJ_BAND_C2 = sprintf('SENS:ACP:OFFSet1:OUT:LIST:BAND:INT %dMHz',ACP_TXCH_BAND);
    fprintf(V, ACP_ADJ_BAND_C2);
    
    % Set adjacent channel spacing
    ACP_ADJ_BAND_SPAC_C = sprintf('SENS:ACP:OFFSet1:OUT:LIST:FREQ %dMHz', ACP_ADJ_BAND_SPAC);
    fprintf(V, ACP_ADJ_BAND_SPAC_C);

    %Set resolution bandwidth
    % Set adjacent channel spacing
    rbw_C = sprintf('SENS:ACP:BAND:RES %dkHz', rbw);
    fprintf(V, rbw_C);


    %set display reference
    ref_lev_C = sprintf('DISP:ACP:VIEW:WIND:TRAC:Y:SCAL:RLEV %d', ref_lev);
    fprintf(V, ref_lev_C);

    % % Measurement only mode
    % % Trigger measurement
    % fprintf(V, 'INIT');
    % 
    % % Wait for measurement to complete
    % pause(1);
    % 
    % % Read ACP measurement results
    % xx = query(V, 'FETCh:ACPower?');
    % x_1 = split(xx, ',');
    P_DBM = 0;
    ACP_L = 0;
    ACP_H = 0;
    
    
    
else
    % Measurement only mode
    % Trigger measurement
    fprintf(V, 'INIT');
    
    % Wait for measurement to complete
    pause(1);
    
    % Read ACP measurement results
    xx = query(V, 'FETCh:ACPower?');
    x_1 = split(xx, ',');
    P_DBM = str2double(string(x_1(2)));
    ACP_L = str2double(string(x_1(5)));
    ACP_H = str2double(string(x_1(7)));
    
end

% Ensure noise correction is enabled
% fprintf(V, 'SENS:POW:NCOR ON');

% Clean up connection
delete(V);

end
