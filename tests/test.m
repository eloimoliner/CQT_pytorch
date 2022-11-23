clear all;
close all;
[a,fs]=audioread("test_dir/0.wav");

a=a(1:131072);

A=fft(a);
N=length(A)

L=512
k=1:L

g=(0.5+0.5*cos(k.*pi*2/L))
H=zeros(length(A),1)
H(1:L/2)=g(1:L/2)
H(N-L/2+1:end)=g(L/2+1:end)

Hhpf=1-H

B=A.*Hhpf
b=real(ifft(B))

C=fft(b)

