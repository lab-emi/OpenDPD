% Part B of the feasibility spike in docs/design/matlab-http-transport.md (not part of the toolbox).
% Measures the MATLAB HTTP client on binary bodies against a throwaway stand-in server (NOT OpenDPD's server), one variant at a
% time: which half of a round trip is slow, and what does the client cost in memory? Linux only (reads /proc for memory).
% Needs REPO_ROOT and SPIKE_STANDIN_PORT.
import matlab.net.http.*
import matlab.net.http.field.*
addpath(fullfile(getenv('REPO_ROOT'), 'Matlab', 'toolbox'));
standinPort = str2double(getenv('SPIKE_STANDIN_PORT'));
standin = sprintf('http://127.0.0.1:%d', standinPort);
opts = HTTPOptions('UseProxy', false, 'ConnectTimeout', 10, 'ResponseTimeout', 900, 'DataTimeout', 900);
tmp = tempname; mkdir(tmp);
cleanup = onCleanup(@() rmdir(tmp, 's'));
fprintf('SPIKE matlab %s\n', version);
accept = HeaderField('Accept', 'application/x-npy');

for logN = [22 24]
    n = 2^logN;
    rng(7);
    x = single(randn(n, 2)) * 0.25;
    megabytes = 8 * n / 2^20;
    npyFile = fullfile(tmp, 'x.npy');
    opendpd.internal.writeNpy(npyFile, x);
    fid = fopen(npyFile, 'r'); raw = fread(fid, Inf, '*uint8'); fclose(fid);
    variants = { ...
        'V1 FileProvider(x-npy)  -> FileConsumer  ', @() fileVariant(npyFile, accept, matlab.net.http.io.FileConsumer(fullfile(tmp, 'out1.npy')), fullfile(tmp, 'out1.npy')); ...
        'V2 MessageBody(uint8)   -> FileConsumer  ', @() struct('request', RequestMessage('POST', [ContentTypeField('application/octet-stream'), accept], MessageBody(raw)), 'consumer', matlab.net.http.io.FileConsumer(fullfile(tmp, 'out2.npy')), 'out', fullfile(tmp, 'out2.npy')); ...
        'V3 MessageBody(uint8)   -> BinaryConsumer', @() struct('request', RequestMessage('POST', [ContentTypeField('application/octet-stream'), accept], MessageBody(raw)), 'consumer', matlab.net.http.io.BinaryConsumer, 'out', ''); ...
        'V4 FileProvider(x-npy)  -> BinaryConsumer', @() fileVariant(npyFile, accept, matlab.net.http.io.BinaryConsumer, '')};
    for v = 1:size(variants, 1)
        if logN == 24 && v == 4, continue; end      % measured before: 80 s
        pieces = variants{v, 2}();
        resetPeak();
        r0 = readStatus('VmRSS');
        tic; [resp, completed] = send(pieces.request, [standin '/infer'], opts, pieces.consumer); t = toc;
        peak = readStatus('VmHWM');
        if isempty(pieces.out)
            bytes = resp.Body.Data(:);
        else
            fid = fopen(pieces.out, 'r'); bytes = fread(fid, Inf, '*uint8'); fclose(fid);
        end
        yy = opendpd.internal.readNpy(bytes, "response");
        maxErr = max(abs(double(yy(:)) - 0.5 * double(x(:))));
        fprintf('SPIKE B N=2^%d (%.0f MiB each way) %s: %6.2f s round trip | extra peak RSS %5d MiB | max error %.2g | status %d\n', ...
            logN, megabytes, variants{v, 1}, t, round((peak - r0) / 1024), maxErr, double(resp.StatusCode));
        clear pieces resp completed yy bytes
    end
    clear raw x
    delete(fullfile(tmp, '*'));
end

% raw uploads: a FileProvider kept in a variable (an inline temporary was deleted by MATLAB before the send in the first run)
payload = fullfile(tmp, 'big.bin');
for mb = [64 512]
    send(RequestMessage('GET'), sprintf('%s/big?mb=%d', standin, mb), opts, matlab.net.http.io.FileConsumer(payload));
    provider = matlab.net.http.io.FileProvider(payload);
    request = RequestMessage('POST', ContentTypeField('application/octet-stream'), provider);
    resetPeak(); r0 = readStatus('VmRSS');
    tic; [resp, completed] = send(request, [standin '/sink'], opts); t = toc;
    fprintf('SPIKE B raw upload %d MiB with FileProvider: %.2f s (%.0f MiB/s) | extra peak RSS %d MiB | server counted %d bytes\n', mb, t, mb / t, round((readStatus('VmHWM') - r0) / 1024), resp.Body.Data.bytes);
    raw = fread(fopen(payload, 'r'), Inf, '*uint8'); fclose('all');
    request = RequestMessage('POST', ContentTypeField('application/octet-stream'), MessageBody(raw));
    resetPeak(); r0 = readStatus('VmRSS');
    tic; [resp, completed] = send(request, [standin '/sink'], opts); t = toc;
    fprintf('SPIKE B raw upload %d MiB with MessageBody(uint8): %.2f s (%.0f MiB/s) | extra peak RSS %d MiB | server counted %d bytes\n', mb, t, mb / t, round((readStatus('VmHWM') - r0) / 1024), resp.Body.Data.bytes);
    clear raw provider request resp completed
    delete(payload);
end

% what one inference-sized call costs when the model is resident: 65,536 samples, best variant (V2), 20 calls
x = single(randn(2^16, 2)) * 0.25;
opendpd.internal.writeNpy(fullfile(tmp, 's.npy'), x);
fid = fopen(fullfile(tmp, 's.npy'), 'r'); raw = fread(fid, Inf, '*uint8'); fclose(fid);
times = zeros(30, 1);
for k = 1:30
    request = RequestMessage('POST', [ContentTypeField('application/octet-stream'), accept], MessageBody(raw));
    tic; send(request, [standin '/infer'], opts, matlab.net.http.io.FileConsumer(fullfile(tmp, 'o.npy'))); times(k) = toc;
end
fprintf('SPIKE B 65,536-sample round trip x30 (V2; the stand-in does 0.5*x): first %.0f ms, median of the rest %.0f ms, min %.0f ms, max %.0f ms\n', 1000 * times(1), 1000 * median(times(2:end)), 1000 * min(times(2:end)), 1000 * max(times(2:end)));
fprintf('SPIKE done\n');

function pieces = fileVariant(file, accept, consumer, out)
import matlab.net.http.*
import matlab.net.http.field.*
provider = matlab.net.http.io.FileProvider(file);
request = RequestMessage('POST', [ContentTypeField('application/x-npy'), accept], provider);
pieces = struct('request', request, 'provider', provider, 'consumer', consumer, 'out', out);     % keep the handle alive
end

function resetPeak()
fid = fopen('/proc/self/clear_refs', 'w'); if fid > 0, fprintf(fid, '5'); fclose(fid); end
end

function value = readStatus(key)
text = fileread('/proc/self/status');
token = regexp(text, [key ':\s+(\d+) kB'], 'tokens', 'once');
value = str2double(token{1});
end
