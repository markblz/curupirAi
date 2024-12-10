library ieee;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity pre_synaptic is
    port(
        clk: in std_logic;
        enable: in std_logic;
        time_to_spike: in integer;
        spike_out: out std_logic
    );
end pre_synaptic;

architecture behave of pre_synaptic is

    signal time_counter: integer := 0;
 
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if (enable = '1') and (time_counter <= time_to_spike + 1) then
                time_counter <= time_counter + 1;
            end if;
        end if;
    end process;
	spike_out <= '1' when (time_counter = time_to_spike) and (enable = '1') else '0';
end behave;