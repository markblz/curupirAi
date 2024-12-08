library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity snn_iris is
    generic (
        delays: integer_vector(40-1 downto 0);
        enable: std_logic_vector(40-1 downto 0)
    );
    port(clk        : in  std_logic; -- input clock 74.25 MHz, video 720p
        ouput_1     : out std_logic;
        ouput_2     : out std_logic;
        ouput_3     : out std_logic
        );                     
end snn_iris;


architecture behave_iris of snn_iris is
    signal spike_link: std_logic_vector(40-1 downto 0);

    component pre_synaptic is
        port(
            clk: in std_logic;
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
            inputs => spike_link,
            output => ouput_1
        );
    target2: entity work.post_synaptic_2
        generic map(N => 40)
        port map(
            clk => clk,
            inputs => spike_link,
            output => ouput_2
        );
    target3: entity work.post_synaptic_3
        generic map(N => 40)
        port map(
            clk => clk,
            inputs => spike_link,
            output => ouput_3
        );
    
    pre_synaptic_all: for i in 0 to 40-1 generate
        pre_synaptic_i: pre_synaptic
            port map(
                clk => clk,
                enable => enable(i),
                time_to_spike => delays(i),
                spike_out => spike_link(i)
            );
    end generate pre_synaptic_all;

end behave_iris;