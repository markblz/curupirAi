library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity neuron_lif is
    generic (
        DATA_WIDTH : integer := 8;   -- Largura dos dados ajustada para 8 bits
        BETA       : integer := 240; -- Valor quantizado de beta (aproximadamente 0.9375 * 256)
        THRESHOLD  : integer := 128  -- Limiar de disparo quantizado (128 para threshold=0.5)
    );
    port (
        clk        : in std_logic;
        reset      : in std_logic;
        input_I    : in integer range -128 to 127;    -- Corrente de entrada ajustada
        mem_pot_out: out integer range -128 to 127;   -- Potencial de membrana atualizado
        spike_out  : out std_logic                    -- Indica se ocorreu um spike
    );
end entity neuron_lif;

architecture Behavioral of neuron_lif is
    signal mem_potential : integer range -128 to 127 := 0;
begin
    process(clk, reset)
    begin
        if reset = '1' then
            mem_potential <= 0;
            spike_out <= '0';
        elsif rising_edge(clk) then
            -- Atualização do potencial de membrana com decaimento
            mem_potential <= ((BETA * mem_potential) + input_I * 256) / 256;  -- Ajuste de escala
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
