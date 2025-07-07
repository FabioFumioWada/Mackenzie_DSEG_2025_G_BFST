/*** Tabela dataset_employess ****/

-- Tabela dataset_employess (Original)
ALTER TABLE dataset_employess ADD ID INT IDENTITY(1,1)
GO
-- Em seguida, defina a coluna como chave primária
ALTER TABLE dataset_employess ADD CONSTRAINT PK_dataset_employess PRIMARY KEY (ID);
GO

-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
SELECT count(*)
FROM dbo.dataset_employess

sp_help dataset_employess

--Attrition - Regra de 3
select	Attrition, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by Attrition


select	OverTime, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by OverTime

select	JobSatisfaction, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by JobSatisfaction

select	Age, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by Age

select	YearsAtCompany, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by YearsAtCompany


	-- Tabela HR (Dataset não utilizado)
	/*
	-- Primeiro, adicione a coluna com propriedade IDENTITY
	ALTER TABLE HR ADD ID INT IDENTITY(1,1)
	GO
	-- Em seguida, defina a coluna como chave primária
	ALTER TABLE HR ADD CONSTRAINT PK_HR PRIMARY KEY (ID);
	GO
	*/


