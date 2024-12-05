library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity neuron_lif is
    generic (
        DATA_WIDTH : integer := 16;  -- Largura dos dados
        BETA       : integer := 15;  -- Valor quantizado de beta (ex: 15 para beta=0.9375)
        THRESHOLD  : integer := 16   -- Limiar de disparo quantizado (ex: 16 para threshold=1.0)
    );
    port (
        clk        : in std_logic;
        reset      : in std_logic;
        input_I    : in integer range -32768 to 32767;  -- Corrente de entrada
        mem_pot_out: out integer range -32768 to 32767;  -- Potencial de membrana atualizado
        spike_out  : out std_logic                      -- Indica se ocorreu um spike
    );
end entity neuron_lif;

architecture Behavioral of neuron_lif is
    signal mem_potential : integer range -32768 to 32767 := 0; -- isso significa que o potencial de membrana é um inteiro de 16 bits
begin
    process(clk, reset)
    begin
        if reset = '1' then
            mem_potential <= 0;
            spike_out <= '0';
        elsif rising_edge(clk) then
            -- Atualização do potencial de membrana com decaimento
            mem_potential <= ((BETA * mem_potential) + input_I) / 16;  -- Dividir por 16 para ajustar a escala
            -- Verificação do limiar
            if mem_potential >= THRESHOLD then
                mem_potential <= 0;  -- Reset do potencial após o spike
                spike_out <= '1';
            else
                spike_out <= '0';
            end if;
        end if;
    end process;
    mem_pot_out <= mem_potential;
end architecture Behavioral;