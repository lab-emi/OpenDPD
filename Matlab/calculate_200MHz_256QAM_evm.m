function evm = calculate_200MHz_256QAM_evm(RX_data, reference_points)
%__author__ = "Yizhuo Wu"
%__license__ = "Apache-2.0 License"
%__email__ = "yizhuo.wu@tudelft.nl"
ff = fftshift(fft(RX_data));
% ff(1:39295) = 0;
% ff(58978:end) = 0;
RX_data = ifft(fftshift(ff));
F_sample = 491.52e6;
f_shift = [-40e6 -20e6 0 20e6 40e6];
start_point = [16929 16922 13207 32235 21522];
input5c = RX_data;
input1c_fd = zeros(5,1212);
input1c_fd_temp = zeros(1,98304);
clean_wv = zeros(5,98304);
MEM = 98304;
EVM5c = zeros(5,1);
for i = 1:5
FinputshiftTimeUP = exp(1i.*(2*pi*([0:MEM-1])/(F_sample/(f_shift(i)))));
input_shift = input5c.*FinputshiftTimeUP;
input_shift_fd = fftshift(fft(input_shift));
% plot(abs(input_shift_fd))
input1c_fd_temp(47351:50995) = input_shift_fd(47351:50995);
clean_wv(i,:) =ifft(fftshift(input1c_fd_temp));
rm_cp = clean_wv(i,start_point(i):start_point(i)+32767);
fd_sym1 = fftshift(fft(rm_cp));
fd_sym1 = fd_sym1./max(abs(fd_sym1));
input1c_fd(i,:) = fd_sym1(15782:16993);
IQsource1 = reference_points(i,:);
IQoutput = input1c_fd(i,:);
IQoutput1 = (IQoutput)./abs(IQoutput).*abs(IQsource1);
% 
EVM5c(i) = 10*log10(sum(((real(IQsource1)-real(IQoutput1)).^2+(imag(IQsource1)-imag(IQoutput1)).^2))/sum(abs(IQoutput1).^2));
end
evm = mean(EVM5c);