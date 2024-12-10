library IEEE;
use IEEE.NUMERIC_STD.ALL;
use IEEE.STD_LOGIC_1164.ALL;

entity clock_divider is
    port (
        clk_in  : in  std_logic;
        clk_out : out std_logic
    );
end clock_divider;

architecture Behavioral_clk of clock_divider is
    signal counter : integer := 0;
    signal clk_out_s : std_logic := '0';
begin
    process(clk_in)
    begin
        if rising_edge(clk_in) then
            if counter = 540000 then
                counter <= 0;
                clk_out_s <= not clk_out_s;
            else
                counter <= counter + 1;
            end if;
        end if;
    end process;

    clk_out <= clk_out_s;

end Behavioral_clk;