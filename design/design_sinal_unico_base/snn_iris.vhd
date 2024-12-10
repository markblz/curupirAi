package mytypes_pkg is

     type my_array_t is array (40-1 downto 0) of integer;
end package mytypes_pkg;

use work.mytypes_pkg.all;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity snn_iris is
    generic (
        delays: my_array_t := (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390);
        enable: std_logic_vector(40-1 downto 0) := (others => '1')
    );
    port(clk        : in  std_logic;
        reset       : in  std_logic; 
        output_1     : out std_logic;
        output_2     : out std_logic;
        output_3     : out std_logic
        );                     
end snn_iris;


architecture behave_iris of snn_iris is
    signal spike_link: std_logic_vector(40-1 downto 0);

    component pre_synaptic is
        port(
            clk: in std_logic;
            reset: in std_logic;
            enable: in std_logic;
            time_to_spike: in natural;
            spike_out: out std_logic
        );
    end component;
    
begin

    target1: entity work.post_synaptic_1
        generic map(N => 40)
        port map(
            clk => clk,
            reset => reset,
            inputs => spike_link,
            output => output_1
        );
    target2: entity work.post_synaptic_2
        generic map(N => 40)
        port map(
            clk => clk,
            reset => reset,
            inputs => spike_link,
            output => output_2
        );
    target3: entity work.post_synaptic_3
        generic map(N => 40)
        port map(
            clk => clk,
            reset => reset,
            inputs => spike_link,
            output => output_3
        );
    
    pre_synaptic_all: for i in 0 to 40-1 generate
        pre_synaptic_i: pre_synaptic
            port map(
                clk => clk,
                reset => reset,
                enable => enable(i),
                time_to_spike => delays(i),
                spike_out => spike_link(i)
            );
    end generate pre_synaptic_all;

end behave_iris;