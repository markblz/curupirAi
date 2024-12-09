library ieee;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity pre_synaptic is
    port(
        clk: in std_logic;
        reset: in std_logic;
        enable: in std_logic;
        time_to_spike: in natural;
        global_time: in integer;
        spike_out: out std_logic
    );
end pre_synaptic;

architecture behave of pre_synaptic is
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if (enable = '1') then
                spike_out <= '1' when (global_time = time_to_spike + 1) else '0'; -- <= time_to_spike + 1
            end if;
        end if;
    end process;

    process(reset)
    begin
        if reset = '1' then
            spike_out <= '0';
        end if;
    end process;
end behave;