library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity synapse is
    generic (
        WEIGHT : integer := 1  -- Peso quantizado
    );
    port (
        spike_in   : in std_logic;
        current_out: out integer range -32768 to 32767
    );
end entity synapse;

architecture Behavioral of synapse is
begin
    current_out <= WEIGHT when spike_in = '1' else 0;
end architecture Behavioral;