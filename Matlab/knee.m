function [knee, k, df1, df2] = knee(x)

df1 = diff(x);
df2 = diff(df1);

k = ((1+df1(1:end-1).^2).^(3/2))./abs(df2(1:end)); % shift 1 to change knee point

[~,knee] = min(k);

end