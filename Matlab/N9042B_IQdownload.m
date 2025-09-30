%__author__ = "Yizhuo Wu"
%__license__ = "Apache-2.0 License"
%__email__ = "yizhuo.wu@tudelft.nl"

function output=N9042B_IQdownload(Ts,fc,fs,Ns)
%
% y=FSPdown(Ns,iter,fc,rel,BW,fs)
%  Ts: time length  of the signal needed (in us)
%  fc: center frequency (in Hz), it can be a vector
%  rel: reference level, it can be a vector
%  BW: resoltion filter bandwidth
%  fs: sampling frequency (in Hz)

V = visa('agilent','GPIB0::18::INSTR');
blocklen=Ns;
NumberOfSamples=Ns;
V.InputBufferSize = blocklen * 4 * 2 + 1;
fopen(V)



 % Configure input attenuation (0 dB)
 % Cpmfigure the measurement mode
fprintf(V, 'INST:CONF:BASIC:WAV');
fprintf(V,'FORMat:BORDer SWAPped');
fprintf(V,'FORMat:TRACe:DATA REAL,32');
 fprintf(V, 'SENS:POW:RF:ATT 10');

% Set center frequency
FREQ_Str = sprintf('SENS:FREQ:CENT %.1f', fc);
fprintf(V, FREQ_Str);

% Set required bandwidth
BW = sprintf('SENS:WAV:SRAT %.1f',fs);
fprintf(V,BW);
% SR = sprintf('SENS:WAV:DIF:BAND %.1f',fs);
% fprintf(V,SR);

%Set required time length
Time_length = sprintf('SENS:WAV:SWE:TIME %dus', Ts);
fprintf(V,Time_length);


fprintf(V,':INIT:IMM')

idn=query(V,'*OPC?');
disp(idn);

fprintf(V,':FETCh:WAVeform0?');

y = binblockread(V,'float32');
pause(10);
I = y(3:2:end);
Q = y(4:2:end);
output = I+1j*Q;

% fprintf(V, 'SENS:POW:RF:ATT 12');
fprintf(V, 'INST:CONF:SA:ACP');
fclose(V);%
delete(V);%
clear V


