library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.ALL;
use std.textio.all;
use ieee.std_logic_textio.all;

entity snn_iris_sim is
end snn_iris_sim;

architecture sim of snn_iris_sim is
    signal clk       : std_logic := '0';
    signal ouput_1s   : std_logic;
    signal ouput_2s   : std_logic;
    signal ouput_3s   : std_logic;
begin

    clk <= not clk after 10 ns;

    dut: entity work.snn_iris
        generic map(
            delays => (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390),
            enable => (others => '1')
        )
        port map(
            clk => clk,
            ouput_1 => output_1s,
            ouput_2 => output_2s,
            ouput_3 => output_3s
        );
end sim;
