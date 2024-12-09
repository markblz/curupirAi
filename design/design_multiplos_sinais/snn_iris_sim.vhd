library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.ALL;
use std.textio.all;
use ieee.std_logic_textio.all;

entity snn_iris_sim is
end snn_iris_sim;

architecture sim of snn_iris_sim is
    signal clk       : std_logic := '0';
    signal reset     : std_logic := '0';
    signal output_1s   : std_logic;
    signal output_2s   : std_logic;
    signal output_3s   : std_logic;
begin

    clk <= not clk after 10 ns;

    dut: entity work.snn_iris
        generic map(
            delays => (0, 0, 393, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0, 393, 751, 0, 0, 0, 0, 0),
            enable => ('0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '0', '0', '0')
        )
        port map(
            clk => clk,
            reset => reset,
            output_1 => output_1s,
            output_2 => output_2s,
            output_3 => output_3s
        );
end sim;
