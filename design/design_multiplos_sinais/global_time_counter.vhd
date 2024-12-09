library ieee;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity global_time_counter is
    port(
        clk: in std_logic;
        reset: in std_logic;
        enable: in std_logic;
        time_counter: out integer
    );
end global_time_counter;

architecture behave of global_time_counter is
    signal local_time_counter: integer := 0;
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if enable = '1' then
                local_time_counter <= local_time_counter + 1;
            end if;
        end if;
    end process;

    process(reset)
    begin
        if reset = '1' then
            local_time_counter <= 0;
        end if;
    end process;

    time_counter <= local_time_counter;
end behave;

# to create and checkout to another branch I need to do: 