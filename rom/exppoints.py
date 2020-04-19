
class ExpPoints:

    def __init__(self):
        pass

    def expansion_freqs(self, num_freqs = ROM.num_freqs , freq_step = 50):
        
        # num_freqs must be a odd number
        if num_freqs%2 == 0:
            num_freqs =+ 1
        
        aux = floor(num_freqs / 2)
        freq_central = median_high(self.freq)
        lower_freq = freq_central - aux * freq_step

        expansion_freqs = np.zeros(num_freqs)

        for i in range(num_freqs):
            freq = i * lower_freq
            index = np.abs(self.freq - freq).argmin()
            expansion_freqs[i] = self.freq[index]
        return expansion_freqs