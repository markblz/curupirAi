library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity post_synaptic_3 is
    generic (
        N : integer := 40  -- Number of inputs, default is 40
    );
    port (
        inputs : in STD_LOGIC_VECTOR(N-1 downto 0); 
        output : out STD_LOGIC 
    );
end entity post_synaptic_3;

architecture Behavioral_3 of post_synaptic_3 is
    signal neuron_input       : integer := 0;
    signal membrane_potential : integer := 0;
    type mem_array is array (0 to N - 1) of integer;
    constant weights : mem_array := (
        25,   25,   25,   25,  262, 1426, 1062, 1568, 1701, 1330,  482, 25, 2223,   25,   25,   25,   25,   25,   25,   25,   25,   25, 25,   25,   25,   25, 1515, 3735, 2514, 2160,   25,   25,   25, 25,   25,   25, 2021, 4057, 3683, 2633
    );
    signal output_enable: STD_LOGIC := '0';
begin
    process(inputs)
    begin
       if output_enable = '0' then
            for i in 0 to N-1 loop
                if inputs(N-1 - i) = '1' then -- alinhar a ordem dos pesos com os inputs
                    neuron_input <= neuron_input + weights(i);
                end if;
            end loop;
            membrane_potential <= membrane_potential +  neuron_input;
	  end if;

            if membrane_potential > 4000 then
                output_enable <= '1';
            else
                output_enable <= '0';
            end if;
    end process;
	
	output <= output_enable;
end architecture Behavioral_3;