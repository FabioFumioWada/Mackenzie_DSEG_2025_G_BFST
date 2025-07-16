/*** Tabela dataset_employess ****/

-- Tabela dataset_employess (Original)
ALTER TABLE dataset_employess ADD ID INT IDENTITY(1,1)
GO
-- Em seguida, defina a coluna como chave prim�ria
ALTER TABLE dataset_employess ADD CONSTRAINT PK_dataset_employess PRIMARY KEY (ID);
GO

-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------

-- Contra Prova

SELECT count(*)
FROM dbo.dataset_employess
go

sp_help dataset_employess
go

select	Attrition, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by Attrition
go


select	OverTime, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by OverTime
go

select	JobSatisfaction, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by JobSatisfaction
go

select	Age, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by Age
go

SELECT MIN(YearsAtCompany), MAX(YearsAtCompany)
from dbo.dataset_employess 
go

SELECT MIN(MonthlyIncome), MAX(MonthlyIncome)
from dbo.dataset_employess 
go

SELECT MIN(DistanceFromHome), MAX(DistanceFromHome)
from dbo.dataset_employess 
go

select	BusinessTravel, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by BusinessTravel
go

select	MaritalStatus, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by MaritalStatus
go


select	Gender, 
		count(*) quantidade, 
		(count(*)*100)/(SELECT count(*) FROM dbo.dataset_employess) percentual
from dbo.dataset_employess 
group by Gender
go


SELECT *
FROM dbo.dataset_employess

SELECT *
FROM dbo.dataset_employess_padronizado



	-- Tabela HR (Dataset n�o utilizado)
	/*
	-- Primeiro, adicione a coluna com propriedade IDENTITY
	ALTER TABLE HR ADD ID INT IDENTITY(1,1)
	GO
	-- Em seguida, defina a coluna como chave prim�ria
	ALTER TABLE HR ADD CONSTRAINT PK_HR PRIMARY KEY (ID);
	GO
	*/


